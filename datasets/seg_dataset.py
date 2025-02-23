import torch
import torchvision
from datasets.utils import read_voc_images, crop_images, colormap
from torch.utils.data import DataLoader


class SegDataset(torch.utils.data.Dataset):
    # 只做了 normalize 和裁剪，没有其他的数据增广
    def __init__(self, voc_path, is_train, crop_size=None):
        super().__init__()
        self.transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        features, labels = read_voc_images(voc_path, is_train)
        features = [self.normalize_img(feature) for feature in features]
        
        # 裁剪图片
        if crop_size is not None:
            h, w = crop_size
            cropped = [crop_images(feature, label, h, w) for (feature, label) in zip(features, labels)]
        else:
            cropped = [crop_images(feature, label) for (feature, label) in zip(features, labels)]
        
        self.features, self.labels = zip(*cropped)
        self.labels = torch.stack(self.labels)  # 将 labels 转换为张量，形状为 [B, 3, H, W]
        
        # 调整 labels 的形状为 [B, H, W, 3]
        self.labels = self.labels.permute(0, 2, 3, 1)  # 将通道维度移到最后
        self.labels = rgb_to_class(self.labels, colormap)  # 将 RGB 标签转换为类别索引
        print(f"read {len(self.labels)} imgs")

    def normalize_img(self, img):
        return self.transform(img.float() / 255)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return (feature, label)
      
    def __len__(self):
        return len(self.labels)

def load_voc(voc_path, crop_size=None, batch_size=64):
    train_iter = DataLoader(SegDataset(voc_path, True, crop_size), batch_size=batch_size, shuffle=True, drop_last=True)
    test_iter = DataLoader(SegDataset(voc_path, False, crop_size), batch_size=batch_size, shuffle=True, drop_last=True)
    return train_iter, test_iter