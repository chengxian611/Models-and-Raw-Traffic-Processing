"""1D CNN model implementation."""
from ..config import Config
import torch
import torch.nn as nn
from .base import BaseModel

class CNN1D(BaseModel):
    """1D CNN model implementation."""
    
    def __init__(self, input_dim=28 * 28, output_dim=128):
        super(CNN1D, self).__init__()
        self.name = "CNN1D"
        oc1=2
        ks = max(int(0.05*input_dim), 3)
        self.conv1 = nn.Conv1d(1, out_channels=oc1, 
                               kernel_size=ks, stride=1, padding=ks//2)
        # self.conv2 = nn.Conv1d(4, out_channels=8, 
        #                        kernel_size=ks, stride=1, padding=ks//2)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        self.relu = nn.ReLU()
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            x = self.relu(self.conv1(dummy_input))
            # x = self.relu(self.conv2(x))
            x = self.pool(x)
            flattened_size = x.view(1, -1).size(1)
        
        self.fc = nn.Linear(flattened_size, output_dim)
        self.fc.register_forward_hook(self._register_hook('after_1dcnn'))
    def forward(self, x):
        if "TA" not in Config.MODEL_NAMES[Config.MODELS_SELECT]:
            x = x.reshape(x.shape[0], -1)
            x = x[:,:Config.d_session]
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x 