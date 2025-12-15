import numpy as np
import pennylane as qml

def encoding(x, wires, threshold=0.0, invert=False):
    """
    Basis (computational) encoding with threshold binarization:
    bits[i] = 1 if x[i] > threshold, 0 otherwise.
    """
    # -> flat numpy
    if hasattr(x, "detach"):            # PyTorch
        z = x.detach().cpu().numpy()
    elif hasattr(x, "numpy"):           # JAX / TF / np
        try:
            z = x.numpy()
        except Exception:
            z = np.array(x)
    else:
        z = np.array(x)

    z = np.ravel(z)
    if len(z) != len(wires):
        raise ValueError(f"len(x)={len(z)} must match len(wires)={len(wires)}")

    bits = (z > threshold).astype(np.int8)
    if invert:
        bits = 1 - bits

    # BasisEmbedding expects a binary vector of length = n_qubits
    qml.BasisEmbedding(features=bits.tolist(), wires=wires)
