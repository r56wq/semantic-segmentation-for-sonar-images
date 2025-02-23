import torch
import torchvision
from datasets.utils import read_voc_images, crop_images, rgb_to_class
from torch.utils.data import DataLoader


class SegDataset(torch.utils.data.Dataset):
    # 只做了 normalize 和裁剪，没有其他的数据增广
    def __init__(self, voc_path, read_type, colormap, crop_size=None):
        super().__init__()
        self.transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.colormap = colormap
        features, labels = read_voc_images(voc_path, colormap, read_type)
        features = [self.normalize_img(feature) for feature in features]
        
        # 裁剪图片
        if crop_size is not None:
            h, w = crop_size
            cropped = [crop_images(feature, label, h, w) for (feature, label) in zip(features, labels)]
        else:
            cropped = [crop_images(feature, label) for (feature, label) in zip(features, labels)]
        
        self.features, self.labels = zip(*cropped) 

    def normalize_img(self, img):
        return self.transform(img.float() / 255)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return (feature, label)
      
    def __len__(self):
        return len(self.labels)

def load_voc(voc_path, colormap, load_type, crop_size=None, batch_size=64):
    if load_type == "training":
        train_iter = DataLoader(SegDataset(voc_path, "train", colormap, crop_size), batch_size=batch_size, shuffle=True, drop_last=True)
        val_iter = DataLoader(SegDataset(voc_path, "val", colormap, crop_size), batch_size=batch_size, shuffle=True, drop_last=True)
        return train_iter, val_iter

    elif load_type == "testing":
        test_iter = DataLoader(SegDataset(voc_path, "test", colormap, crop_size), batch_size=batch_size, drop_last=False)
        return test_iter
    
    else:
        RuntimeError("unexpecting load_type")
