import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from .vqc import QuantumLayer

class HybridNet(nn.Module):
    def __init__(self, n_qubits, n_outputs, n_layers, n_classes, encoding_type, in_channels, freeze_backbone=True):
        super().__init__()
        # Load ResNet-18 pretrained on ImageNet
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze backbone if specified (recommended for comparing encodings)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace FC layer with Identity to get features
        self.backbone.fc = nn.Identity()
        
        # Projection and quantum layers (always trainable)
        self.proj = nn.Linear(512, n_qubits)
        self.q = QuantumLayer(n_qubits=n_qubits, n_outputs=n_outputs, n_layers=n_layers, encoding_name=encoding_type)
        self.head = nn.Linear(n_outputs, n_classes)

    def forward(self, x):
        f = self.backbone(x)
        xq = self.proj(f)
        xq = torch.tanh(xq)
        z = self.q(xq)
        return self.head(z)

class BaselineNet(nn.Module):
    def __init__(self, n_qubits, n_classes, in_channels, freeze_backbone=True):
        super().__init__()
        # Load ResNet-18 pretrained on ImageNet
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace FC layer with Identity
        self.backbone.fc = nn.Identity()
        
        # Projection and classifier (always trainable)
        self.proj = nn.Linear(512, n_qubits)
        self.head = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        f = self.backbone(x)
        xq = self.proj(f)
        return self.head(xq)
