import torch
import torchvision
from .utils import read_voc_images, crop_images
from torch.utils.data import DataLoader

class SegDataset(torch.utils.data.Dataset):
  #只做了normalize, 和裁剪，没有其他的数据增广
    def __init__(self, is_train, voc_path, crop_size = None):
        super().__init__()
        self.transform = torchvision.transforms.Normalize([0, 0, 0], [1, 1, 1])
        features, labels = read_voc_images(voc_path, is_train)
        features = [self.normalize_img(feature) for feature in features]
        #裁剪图片
        if (crop_size is not None):
            h, w = crop_size
            cropped = [crop_images(feature, label, h, w) for (feature, label) in (self.features, self.labels)]
            self.features, self.labels = zip(*cropped)
        else:
            cropped = [crop_images(feature, label) for (feature, label) in (self.features, self.labels)]
            self.features, self.labels = zip(*cropped)
        
        print(f"read {len(labels)} imgs")


    def normalize_img(self, img):
        return self.transform(img.float() / 255)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return (feature, label)
      
    def __len__(self):
        return len(self.labels)
    


def load_voc(voc_path, crop_size = None, batch_size = 64):
    train_iter = DataLoader(SegDataset(voc_path, True, crop_size), batch_size=batch_size, shuffle=True, drop_last=True)
    test_iter = DataLoader(SegDataset(voc_path, False, crop_size), batch_size=batch_size, shuffle=True, drop_last=True)
    return train_iter, test_iter
