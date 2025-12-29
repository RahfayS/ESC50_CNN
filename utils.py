import torch
import torch.nn as nn
from torchvision import models
from model import ResNet34

def load_transfer_model(pth_path, num_classes=50, device=None):
    """
    Loads a transfer-learned ResNet34 for 1-channel input safely.
    
    Args:
        pth_path (str): Path to transfer_model.pth
        num_classes (int): Number of output classes
        device (torch.device): CPU or GPU. If None, auto-detect.
    
    Returns:
        model (nn.Module): Ready-to-use model
    """
    # Detect device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained ResNet34
    model = models.resnet34(weights="IMAGENET1K_V1")
    
    # Modify first conv layer for 1-channel input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    
    # Modify the classifier to match the number of classes of the output 
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load transfer-learned weights safely
    state_dict = torch.load(pth_path, map_location=device)

    # Ignore missing/unexpected keys (like conv1 if it differs)
    model.load_state_dict(state_dict, strict=False)
    
    # Move model to device and set to eval mode
    model.to(device)
    model.eval()
    
    print(f"Pretrained Model Loaded")
    return model, device

def load_scratch_model(pth_path, device = None):
    """
    Loads a from scratch ResNet34
    
    Args:
        pth_path (str): Path to scratch model
        device (torch.device): CPU or GPU. If None, auto-detect.
    
    Returns:
        model (nn.Module): Ready-to-use model
    """
    resNet_scratch = ResNet34() # From Scratch Model
    resNet_scratch.load_state_dict(
        torch.load(pth_path, map_location=device)
    )
    resNet_scratch.to(device)
    print(f"Scratch Model Loaded")
    return resNet_scratch,device