import numpy as np
import torch
from torch import nn

ACTION_SPACE_DIMENSION = 3


class CustomNN(nn.Module):

    def __init__(self):
        super().__init__()

        # CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # MLP
        self.linear = nn.Sequential(
            nn.Linear(16*4*4, 400),
            nn.ReLU(),
            nn.Linear(400, ACTION_SPACE_DIMENSION),
            nn.ReLU()
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x,dtype=torch.float)

        x = self.cnn(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    # Generate a random image
    sample_image = np.random.random((6,1,84,95))
    # Initialise the neural network
    policy = CustomNN()
    # Compute the output
    output = policy(sample_image)
    print("output=",output,output.shape)