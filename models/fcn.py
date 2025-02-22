from models.base_model import interface 
from torch import nn
import torchvision
import torch
from torchvision.transforms.functional import center_crop

class fcn_model(interface):
    def __init__(self, num_classes, crop=(500, 1000)) -> None:
        super().__init__()
        self.num_classes = num_classes 
        self.crop = crop
        pretrained_net = torchvision.models.resnet18(pretrained=False)
        net = nn.Sequential(*list(pretrained_net.children())[:-2])
        num_classes = self.num_classes

        self.conv_final = nn.Conv2d(512, num_classes, kernel_size=1)

        self.upsample = nn.ConvTranspose2d(
            num_classes, num_classes,
            kernel_size=(60, 60),  
            stride=(32, 32),
            padding=(18, 18) 
        )

        self.net = nn.Sequential(
            net,
            self.conv_final,
            self.upsample
        ) 

    def forward(self, x):
        x = self.net(x)
        return center_crop(x, self.crop)

    def save_model(self, path):
        torch.save(self.net.state_dict(), path)

    def load_model(self, path):
        self.net.load_state_dict(torch.load(path))


## dummy test
if __name__ == "__main__":
    x = torch.rand((1, 3, 500, 1000))
    fcn = fcn_model(3)
    y = fcn(x)
    print(y.shape)

