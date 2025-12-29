import torch
import numpy as np
import pandas as pd
import torch.nn as nn
# --- Defining Residual Block For the ResNet34 Model ---
class ResidualBlock(nn.Module):
  """
    Initializes the Residual Block

    Argument:
      in_channels (int): Dimension of input features
      out_channels (int): Dimension of output features
      stride (int): How much the kernel moves across the image
    """
  def __init__(self, in_channels, out_channels, stride=1):
      super().__init__()


      # -- Layer 1 --
      self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
      )
      self.bn1 = nn.BatchNorm2d(out_channels)

      # -- Layer 2 --
      self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
      self.bn2 = nn.BatchNorm2d(out_channels)

      # -- ShortCut --
      # We only use the shortcut when the channels are the same (no downsampling)
      self.apply_shortcut = stride != 1 or in_channels != out_channels
      if self.apply_shortcut:
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size = 1, stride = stride, padding = 0, bias = False),
            nn.BatchNorm2d(out_channels)

          )
      else:
        self.shortcut = None


  def forward(self, x):
      """
        Forward pass of the Residual Block

        Argument:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor after applying the residual block
      """
      shortcut = x

      out = self.conv1(x)
      out = self.bn1(out)
      out = torch.relu(out)

      out = self.conv2(out)
      out = self.bn2(out)

      if self.apply_shortcut:
        shortcut = self.shortcut(x)

      out = out + shortcut

      return torch.relu(out)

class ResNet34(nn.Module):

    def __init__(self,n_classes = 50):
        """
        A ResNet-style Convolutional Neural Network for audio classification using Mel spectrograms.

        Args:
            num_classes (int): Number of output classes (50 for the dataset).
        """
        super().__init__()

        # -- Define Convolutional Layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,stride=2,padding=3,bias = False), # Only 1 input channel for grayscale MEL Spectrogram
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

        )
        # Create the layers of the model containing the residual blocks (refer to resnet diagram).
        self.layer1 = nn.ModuleList([ResidualBlock(64,64) for i in range(3)]) # Creates 3 Residual Blocks
        self.layer2 = nn.ModuleList([ResidualBlock(64 if i == 0 else 128,128,stride=2 if i == 0 else 1) for i in range(4)]) # Creates 4 Residual Blocks
        self.layer3 = nn.ModuleList([ResidualBlock(128 if i ==0 else 256,256,stride=2 if i == 0 else 1) for i in range(6)]) # Creates 6 Residual Blocks
        self.layer4 = nn.ModuleList([ResidualBlock(256 if i == 0 else 512,512,stride=2 if i == 0 else 1) for i in range(3)]) # Creates 3 Residual Blocks

        # Reduce the spatial grid of the 512 channels into a 1x1 value
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(512,n_classes)

    def forward(self,x):
        """
            Forward pass of the entire model

            Argument:
                x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

            Returns:
                torch.Tensor: Output tensor after applying the residual block
        """
        x = self.conv1(x)
        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1) # Flattening the tensor
        x = self.dropout(x)
        x = self.fc(x)

        return x
    



