import numpy as np
import torch
from torch import nn


class CustomNN(nn.Module):

    def __init__(self, output_dimension):
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
            nn.Linear(400, output_dimension),
            nn.ReLU()
        )

    def forward(self, x):
        # Add an extra dimension in case only one image is fed
        if len(x.shape) == 3:
            x = np.reshape(x, (1,) + x.shape)

        if isinstance(x, np.ndarray):
            x = torch.tensor(x,dtype=torch.float)

        x = self.cnn(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    # Generate a random batch of 6 images
    sample_image = np.random.random((6,1,84,95))
    # Initialise the neural network
    ACTION_SPACE_DIMENSION = 3
    policy = CustomNN(output_dimension=ACTION_SPACE_DIMENSION)
    # Compute the output
    output = policy(sample_image)
    print("output=",output,output.shape)