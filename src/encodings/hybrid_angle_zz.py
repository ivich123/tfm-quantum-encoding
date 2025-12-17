import pennylane as qml
import torch

def encoding(x, wires, kappa: float = 1.0, dense: bool = False, angle_scale=None):
    """
    Hybrid Angle+ZZ: RY per qubit + ZZ couplings.
    - x: normalized 1D tensor ~[-1,1]
    - kappa: correlation intensity
    - dense: False -> neighbors; True -> all pairs
    - angle_scale: None->torch.pi (or pass a tensor/float)
    """
    s = torch.pi if angle_scale is None else angle_scale
    n = len(wires)

    # 1) Angle (local information)
    for i, w in enumerate(wires):
        if i < len(x):
            qml.RY(s * x[i], wires=w)

    # 2) ZZ (correlations)
    pairs = ([(i, i+1) for i in range(n-1)]
             if not dense else [(i, j) for i in range(n) for j in range(i+1, n)])
    for i, j in pairs:
        if i < len(x) and j < len(x):
            #phi = kappa * torch.pi * x[i] * x[j]
            phi = kappa * x[i] * x[j]
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ(phi, wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])
