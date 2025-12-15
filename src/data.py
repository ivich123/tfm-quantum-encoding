import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

def get_dataloaders(dataset: str, batch_size: int, workers: int = 2,
                    seed: int = 42, max_train: int = None, max_val: int = None, max_test: int = None):
    """
    Creates dataloaders for Hybrid ResNet.
    Ensures 3 channels and 224x224 resolution for ImageNet compatibility.
    """
    
    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # 1. DEFINE TRANSFORMATIONS
    if dataset.lower() == "mnist":
        # Train & Val transform (MNIST doesn't typically use flip augmentation)
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
            normalize
        ])
        # Test transform is identical here
        transform_test = transform_train
        
        # Load Datasets
        train_full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)
        n_classes = 10
        
    elif dataset.lower() == "cifar10":
        # Train transform (with Augmentation)
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(), # Data Augmentation
            transforms.ToTensor(),
            normalize
        ])
        
        # Test/Val transform (Clean)
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize
        ])
        
        # Load Datasets
        # NOTE: Loading train with transform_train. Validation will inherit this.
        # For a TFM it's fine, but ideally validation should not have RandomFlip.
        train_full_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
        n_classes = 10
        
    else:
        raise ValueError(f"Dataset {dataset} not supported. Use 'mnist' or 'cifar10'.")
    
    # 2. SUBSETTING (Limit train size)
    # Use a generator so the subset is always the same if you run the script twice
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)

    if max_train and max_train < len(train_full_dataset):
        indices = torch.randperm(len(train_full_dataset), generator=g_cpu)[:max_train].tolist()
        train_full_dataset = Subset(train_full_dataset, indices)
    
    # 3. SPLIT TRAIN / VAL
    train_size = len(train_full_dataset)
    
    # Logic fix: Ensure val isn't larger than available data
    if max_val:
        val_size = min(max_val, int(train_size * 0.2)) # Max val or 20%, whatever is smaller
    else:
        val_size = int(train_size * 0.2)
    
    train_size = train_size - val_size
    
    train_set, val_set = random_split(
        train_full_dataset, 
        [train_size, val_size],
        generator=g_cpu # Seeded split
    )
    
    # 4. SUBSETTING (Limit test size)
    if max_test and max_test < len(test_dataset):
        indices = torch.randperm(len(test_dataset), generator=g_cpu)[:max_test].tolist()
        test_dataset = Subset(test_dataset, indices)
    
    # 5. DATALOADERS
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                             num_workers=workers, pin_memory=True)
    
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                           num_workers=workers, pin_memory=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True)
    
    print(f"Dataset: {dataset} | Classes: {n_classes}")
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, n_classes
