import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from utils import load_transfer_model,load_scratch_model
def main():
    # --- Load models ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Define Device

    # Load in Scratch model
    scratch_path = 'models/model_from_scratch.pth'
    resNet_scratch = load_scratch_model(scratch_path,device)

    # Load in transfer learned model
    transferModel_path = 'models/transfer_model.pth'
    resNet_transfer = load_transfer_model(transferModel_path,device=device)


if __name__ == '__main__':
    main()