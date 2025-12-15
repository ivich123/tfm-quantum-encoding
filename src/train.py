import os, time, csv
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import set_seed, get_device, count_trainable_params
from .data import get_dataloaders
from .model import HybridNet, BaselineNet
from .metrics import accuracy, precision_score, get_circuit_specs

def save_row(csv_path, row_dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header: w.writeheader()
        w.writerow(row_dict)

def run_experiment(cfg: dict):
    set_seed(cfg["seed"])
    device = get_device() if cfg.get("device","auto")=="auto" else torch.device(cfg["device"])
    ds = cfg["dataset"]["name"].lower()

    train_loader, val_loader, test_loader, n_classes = get_dataloaders(
        dataset=ds,
        batch_size=cfg["dataset"]["batch_size"],
        workers=cfg["dataset"]["num_workers"],
        max_train=cfg["dataset"]["max_train"],
        max_val=cfg["dataset"]["max_val"],
        max_test=cfg["dataset"]["max_test"],
        seed=cfg["seed"],
    )
    
    freeze_backbone = cfg["model"].get("freeze_backbone", True)

    if cfg["model"]["type"] == "hybrid":
        model = HybridNet(
            n_qubits=cfg["model"]["n_qubits"],
            n_outputs=cfg["model"]["n_outputs"],
            n_layers=cfg["model"].get("n_layers", 1), # Default to 1 if not present
            n_classes=n_classes,
            encoding_type=cfg["model"]["encoding"],
            in_channels=3,  # Always 3 channels with imagenet
            freeze_backbone=freeze_backbone
        )
    else:
        model = BaselineNet(
            n_qubits=cfg["model"]["n_qubits"],
            n_classes=n_classes,
            in_channels=3,  # Always 3 channels with imagenet
            freeze_backbone=freeze_backbone
        )

    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    
    
    # Ensure lr and weight_decay are floats
    lr = float(cfg["optim"]["lr"])
    weight_decay = float(cfg["optim"]["weight_decay"])
    # Optimizer selection
    opt_name = cfg["optim"].get("name", "adam").lower()
    if opt_name == "adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {opt_name} not supported. Use 'adam' or 'sgd'.")

    exp_name = f'{cfg["experiment"]}_{cfg["dataset"]["name"]}_{cfg["model"]["type"]}_{cfg["model"]["encoding"]}'
    csv_path = os.path.join(cfg["log_dir"], "metrics", f"{exp_name}.csv")
    os.makedirs(os.path.join(cfg["log_dir"], "models"), exist_ok=True)

    # Circuit specs will be obtained after the first forward pass
    circuit_specs = {}
    trainable_params = count_trainable_params(model)
    
    # Initial memory info
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        initial_memory_mb = torch.cuda.memory_allocated(device) / 1024**2
    else:
        initial_memory_mb = 0

    best_val, best_path = 0.0, os.path.join(cfg["log_dir"], "models", f"{exp_name}.pt")
    print(f"Device: {device} | Params: {trainable_params:,}")

    for epoch in range(1, int(cfg["optim"]["epochs"])+1):
        t0 = time.time()
        
        # Reset quantum layer stats if it exists
        if cfg["model"]["type"] == "hybrid" and hasattr(model, 'q'):
            model.q.reset_stats()
        
        # --- train ---
        model.train(); tr_loss=tr_acc=tr_prec=0.0; n=0
        quantum_grads = []  # To capture quantum layer gradients
        
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            
            # Capture quantum layer gradient (only for hybrid models)
            if cfg["model"]["type"] == "hybrid" and hasattr(model, 'q') and hasattr(model.q, 'weights'):
                if model.q.weights.grad is not None:
                    quantum_grads.append(model.q.weights.grad.norm().item())
            
            opt.step()
            bs = x.size(0)
            tr_loss += loss.item()*bs
            tr_acc += accuracy(logits,y)*bs
            tr_prec += precision_score(logits, y, average='macro', num_classes=n_classes)*bs
            n+=bs
        tr_loss /= n; tr_acc /= n; tr_prec /= n
        
        # Average quantum layer gradient in this epoch
        avg_quantum_grad = sum(quantum_grads) / len(quantum_grads) if quantum_grads else 0.0
        
        # --- val ---
        model.eval(); va_loss=va_acc=va_prec=0.0; n=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x); loss = crit(logits,y)
                bs = x.size(0)
                va_loss += loss.item()*bs
                va_acc += accuracy(logits,y)*bs
                va_prec += precision_score(logits, y, average='macro', num_classes=n_classes)*bs
                n+=bs
        va_loss/=n; va_acc/=n; va_prec/=n
        dt = time.time()-t0
        
        # Get circuit specs after first epoch (when constructed)
        if epoch == 1 and cfg["model"]["type"] == "hybrid" and hasattr(model, 'q'):
            n_layers = cfg["model"].get("n_layers", 1)
            circuit_specs = get_circuit_specs(model.q.circuit, cfg["model"]["encoding"], n_layers=n_layers)
            if circuit_specs:
                print(f"Circuit: {circuit_specs['circuit_n_qubits']} qubits | Depth: {circuit_specs['circuit_depth']} | Gates: {circuit_specs['total_gates']} (CNOTs: {circuit_specs['cnot_gates']})")
                print()
        
        # Average quantum circuit time (encoding + ansatz)
        quantum_time_ms = 0.0
        if cfg["model"]["type"] == "hybrid" and hasattr(model, 'q'):
            quantum_time_ms = model.q.get_avg_time_per_sample()
        
        # Memory metrics
        if device.type == 'cuda':
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
            current_memory_mb = torch.cuda.memory_allocated(device) / 1024**2
        else:
            peak_memory_mb = 0
            current_memory_mb = 0

        row = dict(
            epoch=epoch,
            train_loss=tr_loss,
            train_acc=tr_acc,
            train_precision=tr_prec,
            val_loss=va_loss,
            val_acc=va_acc,
            val_precision=va_prec,
            epoch_time_s=dt,
            quantum_circuit_time_ms=quantum_time_ms,
            quantum_grad_norm=avg_quantum_grad,  # Average quantum gradient norm
            dataset=ds,
            model=cfg["model"]["type"],
            encoding=cfg["model"]["encoding"],
            n_qubits=cfg["model"]["n_qubits"],
            n_layers=cfg["model"].get("n_layers", 1),
            n_outputs=cfg["model"]["n_outputs"],
            trainable_params=trainable_params,
            peak_memory_mb=peak_memory_mb,
            current_memory_mb=current_memory_mb,
            **circuit_specs  # Add circuit specs
        )
        save_row(csv_path, row)
        print(f"[{epoch:03d}] tr {tr_loss:.4f}/{tr_acc:.3f} | va {va_loss:.4f}/{va_acc:.3f} | {dt:.1f}s")

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), best_path)

    # test final
    te_loss=te_acc=te_prec=0.0; n=0
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            logits = model(x); loss = crit(logits,y)
            bs = x.size(0)
            te_loss += loss.item()*bs
            te_acc += accuracy(logits,y)*bs
            te_prec += precision_score(logits, y, average='macro', num_classes=n_classes)*bs
            n+=bs
    te_loss/=n; te_acc/=n; te_prec/=n
    
    test_row = dict(
        epoch="test",
        train_loss="",
        train_acc="",
        train_precision="",
        val_loss="",
        val_acc="",
        val_precision="",
        epoch_time_s="",
        quantum_circuit_time_ms="",
        test_loss=te_loss,
        test_acc=te_acc,
        test_precision=te_prec,
        dataset=ds,
        model=cfg["model"]["type"],
        encoding=cfg["model"]["encoding"],
        n_qubits=cfg["model"]["n_qubits"],
        n_outputs=cfg["model"]["n_outputs"],
        trainable_params=trainable_params,
        peak_memory_mb=peak_memory_mb if device.type == 'cuda' else 0,
        current_memory_mb=current_memory_mb if device.type == 'cuda' else 0,
        **circuit_specs
    )
    save_row(csv_path, test_row)
    print(f"TEST: loss {te_loss:.4f} acc {te_acc:.3f} prec {te_prec:.3f} | best val {best_val:.3f}")
