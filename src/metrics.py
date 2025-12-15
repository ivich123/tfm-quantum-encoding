import torch
import pennylane as qml


def accuracy(logits, targets):
    """
    Calculates accuracy given logits and targets.
    
    Args:
        logits: tensor of shape (batch_size, n_classes)
        targets: tensor of shape (batch_size,)
    
    Returns:
        float: accuracy in the range [0, 1]
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


def top_k_accuracy(logits, targets, k=5):
    """
    Calculates top-k accuracy.
    
    Args:
        logits: tensor of shape (batch_size, n_classes)
        targets: tensor of shape (batch_size,)
        k: int, number of top predictions to consider
    
    Returns:
        float: top-k accuracy in the range [0, 1]
    """
    _, top_k_preds = logits.topk(k, dim=1, largest=True, sorted=True)
    targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).sum().item()
    total = targets.size(0)
    return correct / total


def get_circuit_specs(circuit, encoding_name=None, n_layers=1):
    """
    Retrieves quantum circuit specifications by extracting real operations.
    
    Args:
        circuit: PennyLane QNode
        encoding_name: Name of the encoding used (for debug)
        n_layers: Number of ansatz layers
        
    Returns:
        dict with circuit_n_qubits, circuit_depth, total_gates, cnot_gates
    """
    try:
        # Get number of qubits (compatible with new and legacy devices)
        if hasattr(circuit.device, 'num_wires'):
            n_qubits = circuit.device.num_wires
        elif hasattr(circuit.device, 'wires'):
            n_qubits = len(circuit.device.wires)
        else:
            n_qubits = len(circuit.device._wires)
        
        # Build tape with dummy data to extract operations
        # IMPORTANT: Do not use zeros because amplitude encoding requires norm != 0
        dummy_inputs = torch.randn(n_qubits)
        # Weights with shape (n_layers, n_qubits)
        dummy_weights = torch.randn(n_layers, n_qubits)
        
        # Construct the circuit and get the tape
        tape = circuit.construct((dummy_inputs, dummy_weights), {})
        
        # Extract operations from tape
        ops = tape.operations
        total_gates = len(ops)
        
        # Count CNOTs and other 2-qubit gates
        cnot_gates = sum(1 for op in ops if op.name in ['CNOT', 'CZ', 'CX'])
        
        # Calculate circuit depth
        depth = calculate_circuit_depth(ops)
        
        return {
            'circuit_n_qubits': n_qubits,
            'circuit_depth': depth,
            'total_gates': total_gates,
            'cnot_gates': cnot_gates
        }
    except Exception as e:
        print(f"⚠️  Error extracting circuit specs: {e}")
        return {}


def calculate_circuit_depth(operations):
    """
    Calculates circuit depth based on operations.
    Depth = maximum number of layers where gates must be executed sequentially.
    
    Args:
        operations: List of PennyLane operations
        
    Returns:
        int: Circuit depth
    """
    if not operations:
        return 0
    
    # Track which layer each qubit is available in
    qubit_layers = {}
    max_depth = 0
    
    for op in operations:
        # Get qubits used by this operation
        wires = [int(w) for w in op.wires]
        
        # Find the earliest layer where all qubits are available
        earliest_layer = max([qubit_layers.get(w, 0) for w in wires])
        
        # This operation executes at earliest_layer + 1
        current_layer = earliest_layer + 1
        
        # Update layers for used qubits
        for w in wires:
            qubit_layers[w] = current_layer
        
        # Update max depth
        max_depth = max(max_depth, current_layer)
    
    return max_depth


def precision_score(logits, targets, average='macro', num_classes=None):
    """
    Calculates precision for multiclass classification.
    
    Args:
        logits: tensor of shape (batch_size, n_classes)
        targets: tensor of shape (batch_size,)
        average: 'macro', 'micro' or 'weighted'
        num_classes: number of classes (auto-detected if None)
        
    Returns:
        float: precision score
    """
    preds = torch.argmax(logits, dim=1)
    
    if num_classes is None:
        num_classes = logits.shape[1]
    
    if average == 'micro':
        # Micro: count all TPs and FPs globally
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return correct / total if total > 0 else 0.0
    
    precisions = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        
        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
        else:
            precisions.append(0.0)
    
    if average == 'macro':
        return sum(precisions) / len(precisions) if precisions else 0.0
    elif average == 'weighted':
        # Weight by support (number of samples) of each class
        weights = [(targets == c).sum().item() for c in range(num_classes)]
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        return sum(p * w for p, w in zip(precisions, weights)) / total_weight
    
    return precisions
