import torch.nn as nn
from abc import ABC, abstractmethod

class interface(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def save_model(self, path):
        pass
