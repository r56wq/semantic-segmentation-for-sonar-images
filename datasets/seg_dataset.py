import torch
import torchvision
from datasets.utils import read_voc_images, crop_images, colormap
from torch.utils.data import DataLoader

def rgb_to_class(labels, colormap):
    """
    将 RGB 标签图像转换为类别索引图像。
    
    参数:
        labels (torch.Tensor): RGB 标签图像，形状为 [B, H, W, 3]。
        colormap (torch.Tensor): 颜色映射表，形状为 [C, 3]，C 是类别数。
    
    返回:
        torch.Tensor: 类别索引图像，形状为 [B, H, W]。
    """
    assert len(labels.shape) == 4 and labels.shape[3] == 3, "labels 的形状应为 [B, H, W, 3]"
    assert len(colormap.shape) == 2 and colormap.shape[1] == 3, "colormap 的形状应为 [C, 3]"
    
    # 将 labels 和 colormap 转换为相同的形状以便比较
    labels = labels.unsqueeze(3)  # [B, H, W, 1, 3]
    colormap = colormap.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, C, 3]
    
    # 计算每个像素与 colormap 的匹配情况
    matches = (labels == colormap).all(dim=-1)  # [B, H, W, C]
    
    # 找到每个像素匹配的类别索引
    class_indices = matches.long().argmax(dim=-1)  # [B, H, W]
    
    # 检查是否有未匹配的像素
    unmatched = ~matches.any(dim=-1)
    if unmatched.any():
        # 打印未匹配的像素值及其位置
        unmatched_indices = torch.nonzero(unmatched)  # 找到未匹配像素的索引
        for idx in unmatched_indices:
            b, h, w = idx.tolist()  # 批次、高度、宽度索引
            pixel_value = labels[b, h, w, 0].tolist()  # 未匹配的像素值
            print(f"未匹配的像素值: {pixel_value}，位置: (批次={b}, 行={h}, 列={w})")
        raise ValueError("存在未匹配的像素值，请检查标签图像或颜色映射表。")
    
    return class_indices

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