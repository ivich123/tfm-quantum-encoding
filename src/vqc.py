import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from .encodings import get_encoding

def _pl_device(n_qubits: int):
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
        print("✓ Using lightning.gpu (GPU)")
        return dev
    except Exception:
        pass
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        print("✓ Using lightning.qubit (Optimized CPU)")
        return dev
    except Exception:
        pass
    try:
        dev = qml.device("lightning.kokkos", wires=n_qubits)
        print("✓ Using lightning.kokkos (Kokkos)")
        return dev
    except Exception:
        pass
    print("⚠️  Using default.qubit (Standard CPU)")
    return qml.device("default.qubit", wires=n_qubits)

def _pick_diff_method(dev):
    # adjoint is faster and more stable on statevector CPU (default.qubit, lightning.qubit)
    # On lightning.gpu it is often unavailable: falls back to parameter-shift.
    name = getattr(dev, "short_name", "") or getattr(dev, "name", "")
    if "lightning.qubit" in name or "default.qubit" in name:
        return "adjoint"
    return "parameter-shift"

def make_qnode(n_qubits: int, n_outputs: int, n_layers: int, encoding_name: str):
    assert 1 <= n_outputs <= n_qubits, "n_outputs must be in [1, n_qubits]"
    enc_fn = get_encoding(encoding_name)
    dev = _pl_device(n_qubits)
    diff_method = _pick_diff_method(dev)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        # 1) Encoding (encode classical data into quantum state)
        enc_fn(inputs, wires=range(n_qubits))
        
        # 2) Variational Layers (Ansatz)
        # weights shape: (n_layers, n_qubits)
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[l, i], wires=i)
            # Entanglement (ring topology)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            # Optional: Close the ring for better entanglement if n_qubits > 2
            # if n_qubits > 2: qml.CNOT(wires=[n_qubits-1, 0])
        
        # 3) Measurement: PauliZ expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(n_outputs)]

    return circuit

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits: int, n_outputs: int, n_layers: int, encoding_name: str):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.encoding_name = encoding_name

        self.circuit = make_qnode(n_qubits, n_outputs, n_layers, encoding_name)

        # Ansatz weights: shape (n_layers, n_qubits)
        # Moderate initialization to break symmetry
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.5)

        self.total_samples = 0
        self.total_time = 0.0

    def forward(self, x):
        # x: (batch, n_qubits). Loop kept for simplicity/compatibility.
        import time
        batch_size = x.shape[0]
        
        #TIMER: Measure GPU→CPU transfer
        t_transfer_start = time.perf_counter()
        x_cpu = x.detach().cpu() if x.is_cuda else x
        t_transfer_gpu_to_cpu = time.perf_counter() - t_transfer_start
        
        results = []
        t0 = time.perf_counter()
        for i in range(batch_size):
            result = self.circuit(x_cpu[i], self.weights)  # list of n_outputs
            results.append(torch.stack(result))
        t_circuit = time.perf_counter() - t0
        
        #TIMER: Measure CPU→GPU transfer
        t_transfer_start = time.perf_counter()
        output = torch.stack(results).float()
        if x.is_cuda:
            output = output.cuda()
        t_transfer_cpu_to_gpu = time.perf_counter() - t_transfer_start
        
        #Print timing breakdown (only first batch of each epoch)
        if self.total_samples == 0:
            print(f"\n[VQC Timing - {self.encoding_name}]")
            print(f"  GPU→CPU transfer: {t_transfer_gpu_to_cpu*1000:.2f} ms")
            print(f"  Circuit execution: {t_circuit*1000:.2f} ms ({batch_size} samples)")
            print(f"  CPU→GPU transfer: {t_transfer_cpu_to_gpu*1000:.2f} ms")
            print(f"  Total overhead:   {(t_transfer_gpu_to_cpu + t_transfer_cpu_to_gpu)*1000:.2f} ms")
        
        self.total_samples += batch_size
        self.total_time += t_circuit
        return output

    def get_avg_time_per_sample(self):
        if self.total_samples == 0:
            return 0.0
        return (self.total_time / self.total_samples) * 1000

    def reset_stats(self):
        self.total_samples = 0
        self.total_time = 0.0
