import pennylane as qml
from math import pi

def encoding(x, wires):
    # Scale inputs so rotations are more significant
    # Projected ResNet features are usually in [-1, 1], scale to [-π, π]
    for i, w in enumerate(wires):
        qml.RY(pi * x[i], wires=w)
