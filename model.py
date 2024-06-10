import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN_CNN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(DQN_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=32, stride=8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=16, stride=4)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=8, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=1)
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_w_1 = conv2d_size_out(input_shape[1], 32, 8)
        conv_h_1 = conv2d_size_out(input_shape[2], 32, 8)
        conv_w_2 = conv2d_size_out(conv2d_size_out(input_shape[1], 32, 8), 16, 4)
        conv_h_2 = conv2d_size_out(conv2d_size_out(input_shape[2], 32, 8), 16, 4)
        conv_w_3 = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 32, 8), 16, 4), 8, 2)
        conv_h_3 = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 32, 8), 16, 4), 8, 2)
        conv_w_4 = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 32, 8), 16, 4), 8, 2), 4, 1)
        conv_h_4 = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 32, 8), 16, 4), 8, 2), 4, 1)
        
        self.linear_input_size = (conv_w_4 * conv_h_4 * 128) + (conv_w_3 * conv_h_3 * 64) + (conv_w_2 * conv_h_2 * 64) + (conv_w_1 * conv_h_1 * 32)

        # Reduce the size of the fully connected layers
        self.fc1 = nn.Linear(self.linear_input_size, 256)  # Adjust input size accordingly
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x4))

        # Flatten each conv layer output
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)
        x3_flat = x3.view(x3.size(0), -1)
        x4_flat = x4.view(x3.size(0), -1)

        # Concatenate flattened outputs
        x = torch.cat((x1_flat, x2_flat, x3_flat, x4_flat), dim=1)

        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x