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
