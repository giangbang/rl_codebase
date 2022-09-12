import torch 
import torch.nn as nn

class MLP(nn.Module):
    '''
    Multi-layer perceptron.

    :param inputs_dim: 1D Input dimension of the input data
    :param outputs_dim: output dimension
    :param n_layer: total number of layer in MLP, minimum is two
    :param n_unit: dimensions of hidden layers 
    '''
    def __init__(self, inputs_dim: int, outputs_dim:int, n_layer:int, n_unit:int=256):
        super().__init__()
        self.inputs_dim     = inputs_dim
        self.outputs_dim    = outputs_dim
        
        net = [nn.Linear(inputs_dim, n_unit), nn.ReLU()]
        for _ in range(n_layer-2):
            net.append(nn.Linear(n_unit, n_unit))
            net.append(nn.ReLU())
        net.append(nn.Linear(n_unit, outputs_dim))
        
        self.net = nn.Sequential(*net)
        
    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    The code is taken from stable-baselines3, with some modifications.
    """
    def __init__(self, n_input_channels: int, features_dim: int=512):
        super().__init__()
        from torchvision.transforms import Resize
        self.cnn = nn.Sequential(
            Resize((84, 84)), # input image is resized to 84x84
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0), # 32x20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), # 64x9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # 64x7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, features_dim)
        )

    def forward(self, obs):
        obs = (obs-128)/128 # normalize input
        return self.cnn(obs)