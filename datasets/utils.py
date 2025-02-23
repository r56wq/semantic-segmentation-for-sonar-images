import os
import torchvision

import os
import torch
import torchvision
from collections import defaultdict

colormap = torch.tensor([[0, 0, 0], [250, 50, 83], [51, 221, 255]], dtype=torch.uint8)
classes = ["background", "plane", "boat"]

def read_voc_images(voc_path, is_train=True):
    """
    从 VOC 数据集中读取图像和标签。
    
    参数:
        voc_path (str): VOC 数据集的根目录。
        is_train (bool): 是否为训练模式。如果为 True，读取训练集；否则读取测试集。
    
    返回:
        features (list): 图像列表，每个图像是一个 torch.Tensor。
        labels (list): 标签列表，每个标签是一个 torch.Tensor。
    """
    mode = torchvision.io.image.ImageReadMode.RGB
    Annotations_path = os.path.join(voc_path, "train" if is_train else "test", "Annotations")
    JPEGImages_path = os.path.join(voc_path, "train" if is_train else "test", "JPEGImages")
    JPEGImages_names = os.listdir(JPEGImages_path)
    
    features, labels = [], []
    
    for name in JPEGImages_names:
        # 读取图像
        feature = torchvision.io.read_image(os.path.join(JPEGImages_path, name), mode)
        features.append(feature)
        
        # 读取标签
        label = torchvision.io.read_image(os.path.join(Annotations_path, name.split(".")[0] + ".png"), mode)
        labels.append(label)
         
    return features, labels

def crop_images(feature, label, height = 500, width = 1000):
    # CenterCrop 会自动补0如果图片太小了，所以对feature和label都可以用
    centercrop = torchvision.transforms.CenterCrop((height, width))
    feature = centercrop(feature)
    label = centercrop(label)
    return feature, label


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
