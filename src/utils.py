import random
import numpy as np
import torch


def set_seed(seed: int):
    """Sets seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism (might be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Automatically detects if CUDA is available and functional."""
    if torch.cuda.is_available():
        try:
            # Functional test: Try to allocate a tensor on GPU and perform a computation
            # This catches "cudaErrorNoKernelImageForDevice" which happens when 
            # PyTorch version is incompatible with the GPU architecture.
            dummy = torch.zeros(1).cuda()
            _ = dummy * 2  # Actual kernel execution
            del dummy
            
            device = torch.device("cuda")
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"⚠ GPU detected but failed functional test: {e}")
            print("⚠ Fallback to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("⚠ CUDA not available, using CPU")
    return device


def count_trainable_params(model):
    """Counts the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    """Counts the total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def freeze_model(model):
    """Freezes all model parameters."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    """Unfreezes all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
