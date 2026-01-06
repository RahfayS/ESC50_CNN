import torch
import torch.nn as nn
from torchvision import models
from model import ResNet34

def load_model(pth_path, device = None):
    """
    Loads a from scratch ResNet34
    
    Args:
        pth_path (str): Path to scratch model
        device (torch.device): CPU or GPU. If None, auto-detect.
    
    Returns:
        model (nn.Module): Ready-to-use model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resNet = ResNet34() # From Scratch Model
    resNet.load_state_dict(
        torch.load(pth_path, map_location=device)
    )
    resNet.to(device)
    print(f"Model Loaded")
    resNet.eval()
    return resNet