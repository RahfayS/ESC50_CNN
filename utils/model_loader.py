import torch
from model import ResNet34
from huggingface_hub import hf_hub_download
import streamlit as st

@st.cache_resource(show_spinner="Loading CNN models...")
def load_model(device = None):
    """
    Loads a from scratch ResNet34
    
    Args:
        pth_path (str): Path to scratch model
        device (torch.device): CPU or GPU. If None, auto-detect.
    
    Returns:
        models (nn.Module): Ready-to-use model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resNet_balanced = ResNet34() # From Scratch Model
    resNet_balanced_path = hf_hub_download(
        repo_id="Rahfay/ESC-50-CNN",
        filename="ResNet34_balanced_augmentation.pth"
    )

    resNet_acoustic = ResNet34() # From Scratch Model
    resNet_acoustic_path = hf_hub_download(
        repo_id="Rahfay/ESC-50-CNN",
        filename="ResNet34_acoustic_augmentation.pth"
    )

    resNet_balanced.load_state_dict(torch.load(resNet_balanced_path,map_location=device))
    resNet_acoustic.load_state_dict(torch.load(resNet_acoustic_path,map_location=device))

    return resNet_balanced, resNet_acoustic