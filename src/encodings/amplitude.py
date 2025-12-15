import pennylane as qml
import torch
import torch.nn.functional as F

def encoding(x, wires):
    """
    Amplitude encoding with NaN/Inf cleaning and internal normalization.
    - Uses ALL qubits: dim = 2**n_qubits
    - Letting AmplitudeEmbedding handle normalization (normalize=True)
    """
    n = len(wires)
    dim = 2 ** n

    # Ensure 1D tensor and stable dtype for PennyLane
    v = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    v = v.flatten().to(dtype=torch.float64)

    # Replace NaN/Inf with 0.0 (avoid NaN norm)
    v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    # Pad/truncate to dim
    d = v.shape[0]
    if d < dim:
        v = F.pad(v, (0, dim - d), value=0.0)
    elif d > dim:
        v = v[:dim]

    # If everything is zero, use |0...0> to avoid normalization issues
    if torch.all(v == 0):
        v = v.clone()
        v[0] = 1.0  # valid basis state; PL normalizes it anyway
    
    # Let PL normalize internally (safer)
    qml.AmplitudeEmbedding(features=v, wires=wires, pad_with=0.0, normalize=True)
