import os
import torchvision

import os
import torch
import torchvision
from collections import defaultdict
import matplotlib.pyplot as plt

colormap = [[0, 0, 0], [250, 50, 83], [51, 221, 255]]
classes = ["background", "plane", "boat"]

import torch

def image_to_class_indices(raw_label, colormap):
    """
    Convert an RGB label image to a class index image.
    
    Args:
        raw_label (torch.Tensor): RGB label image, shape [C, H, W] (e.g., [3, H, W]).
        colormap (torch.Tensor): Color map, shape [num_classes, 3].
    
    Returns:
        torch.Tensor: Class index image, shape [H, W].
    """
    # Ensure raw_label is [H, W, 3] by permuting if needed (from [3, H, W])
    if raw_label.shape[0] == 3:
        raw_label = raw_label.permute(1, 2, 0)  # [H, W, 3]

    H, W, _ = raw_label.shape
    labels_flat = raw_label.reshape(-1, 3)  # [H*W, 3]

    # Convert colormap to tensor if it isn’t already
    colormap = torch.tensor(colormap, device=raw_label.device, dtype=raw_label.dtype)  # [C, 3]

    # Compare each pixel’s RGB with colormap rows: [H*W, C]
    matches = (labels_flat.unsqueeze(1) == colormap.unsqueeze(0)).all(dim=2)  # [H*W, C], boolean

    # Convert boolean tensor to integer (True -> 1, False -> 0)
    matches = matches.to(torch.uint8)  # or matches.long(), matches.float()

    # Get class indices: [H*W]
    class_indices = matches.argmax(dim=1)  # Now works because matches is numeric

    # Reshape back to [H, W]
    return class_indices.reshape(H, W)

def read_voc_images(voc_path, colormap, is_train=True):
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
        raw_label = torchvision.io.read_image(os.path.join(Annotations_path, name.split(".")[0] + ".png"), mode)
        label = image_to_class_indices(raw_label, colormap)
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
    assert len(colormap.shape) == 2 and colormap.shape[1] == 3, "colormap的形状应为[C, 3]"

    # Reshape labels to [B*H*W, 3] for broadcasting
    B, H, W, _ = labels.shape
    labels_flat = labels.reshape(-1, 3)  # [B*H*W, 3]

    # Ensure colormap is a tensor and reshape for broadcasting
    colormap = torch.tensor(colormap, device=labels.device, dtype=labels.dtype)  # [C, 3]

    # Compare each pixel's RGB with colormap rows: [B*H*W, C]
    matches = (labels_flat.unsqueeze(1) == colormap.unsqueeze(0)).all(dim=2)  # [B*H*W, C]

    # Get class indices: [B*H*W]
    class_indices = matches.argmax(dim=1)  # If no match, argmax gives 0 (background)

    # Reshape back to [B, H, W]
    return class_indices.reshape(B, H, W)


def plot_images(features):
    # features (batch_size, 3, h, w )
    # 没有标签
    plt.figure(figsize=(15, 5))
    for i in range(len(features)):
        img = features[i].permute(1, 2, 0).numpy()
        plt.subplot(1, len(features), i+1)
        plt.imshow(img)
    plt.show()


def labels_to_rgb(labels):
    #labels (batch_size, h, w)
    # 把labels转换为RGB图像， 根据colormap
    labels = labels.unsqueeze(3)  # [B, H, W, 1, 3]
    colormap = colormap.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, C, 3]
    matches = (labels == colormap).all(dim=-1)  # [B, H, W, C]
    class_indices = matches.long().argmax(dim=-1)  # [B, H, W]
    return class_indices
