import pennylane as qml
from math import pi

def encoding(x, wires, kappa=1.0):
    # Feature map based on ZZ-feature map from Havlicek et al.
    # First apply Hadamard to create superposition
    for w in wires:
        qml.Hadamard(wires=w)
    
    # First layer of Z rotations
    for i, w in enumerate(wires):
        qml.RZ(pi * x[i], wires=w)
    
    # Entanglement with ZZ terms
    for i in range(len(wires)-1):
        w1, w2 = wires[i], wires[i+1]
        phi = kappa * x[i] * x[i+1]  # No extra pi to avoid very small products
        qml.CNOT(wires=[w1, w2])
        qml.RZ(phi, wires=w2)
        qml.CNOT(wires=[w1, w2])
