import os
import torchvision

import os
import torch
import torchvision
from collections import defaultdict
import matplotlib.pyplot as plt


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

def read_voc_images(voc_path, colormap, read_type):
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
    if (read_type == "train"): 
        Annotations_path = os.path.join(voc_path, "train", "Annotations")
        JPEGImages_path = os.path.join(voc_path, "train", "JPEGImages")

    elif (read_type == "val"):
        Annotations_path = os.path.join(voc_path, "val", "Annotations")
        JPEGImages_path = os.path.join(voc_path, "val", "JPEGImages")

    elif (read_type == "test"):
        Annotations_path = os.path.join(voc_path, "test", "Annotations")
        JPEGImages_path = os.path.join(voc_path, "test", "JPEGImages")
    else:
        RuntimeError("invalid read_type")
        
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
    fig = features.int()
    plt.figure(figsize=(15, 5))
    for i in range(len(fig)):
        img = fig[i].permute(1, 2, 0).numpy()
        plt.subplot(1, len(fig), i+1)
        plt.imshow(img)
    plt.show()


def labels_to_rgb(labels, colormap):
    # labels (batch_size, h, w)
    # 把labels转换为RGB图像， 根据colormap
    B, H, W = labels.shape
    rgb_images = torch.zeros((B, H, W, 3), dtype=torch.uint8, device=labels.device)
    for i in range(len(colormap)):
        rgb_images[labels == i] = colormap[i]
    return rgb_images

def set_value(picture, value, x, y):
        """
        设置图片的某个像素点的值 
    参数:
        picture (torch.Tensor): 图像，形状为 [3, H, W]。
        value(List): 一个list，表示 RGB 值。
        x (int): 行坐标。
        y (int): 列坐标。
    """
        picture[0][x][y] = value[0]
        picture[1][x][y] = value[1]
        picture[2][x][y] = value[2]

def classes_to_pic(class_indices, colormap):
        """
        设置图片的某个像素点的值 
    参数:
        classs_indices (torch.Tensor): (H, W) 表示类别的一个张量
        colormap (List): 表示颜色和类别的映射
    返回:
        类别对应的RGB图像
    """
        H, W = class_indices.shape
        pic = torch.zeros((3, H, W))
        for h in range(H):
            for w in range(W):
                class_id = class_indices[h][w]
                pic[0][h][w] = colormap[class_id][0]
                pic[1][h][w] = colormap[class_id][1]
                pic[2][h][w] = colormap[class_id][2]
        return pic                

import torch

def print_non_zeros(img: torch.Tensor):
    """
    打印一个张量中不为0的元素，如果通道数为3，则打印RGB值
    参数:
        img (torch.Tensor): 图像，形状为 [3, H, W] 或者 [1, H, W]
    """
    # 检查输入张量的维度
    assert img.dim() == 3, "Input tensor must have 3 dimensions [C, H, W]"
    channels, height, width = img.shape
    
    # 确保通道数为1或3
    assert channels in [1, 3], "Channels must be 1 or 3"
    
    # 找到非零元素的位置
    nonzero_indices = torch.nonzero(img, as_tuple=False)
    
    if nonzero_indices.numel() == 0:
        print("图像中没有非零元素")
        return
    
    # 遍历非零位置并打印值
    print(f"非零元素 (通道数={channels}):")
    for idx in nonzero_indices:
        c, h, w = idx[0], idx[1], idx[2]  # 通道、高度、宽度坐标
        if channels == 3:
            # RGB图像，打印RGB值
            r, g, b = img[:, h, w]
            print(f"位置 ({h}, {w}): RGB = ({r.item()}, {g.item()}, {b.item()})")
        else:
            # 单通道图像，打印单一值
            value = img[0, h, w]
            print(f"位置 ({h}, {w}): 值 = {value.item()}")


