import os
import torchvision

def read_voc_images(voc_path, is_train = True):
    # read images from voc_dir, if train mode is set, read from the train directory, otherwise
    # read from val directory
    mode = torchvision.io.image.ImageReadMode.RGB
    Annotations_path = os.path.join(voc_path, "train" if is_train else "test", "Annotations")
    JPEGImages_path = os.path.join(voc_path, "train" if is_train else "test", "JPEGImages")
    #Annotation_names = os.listdir(Annotations_path)
    JPEGImages_names = os.listdir(JPEGImages_path)
    features, labels = [], []
    for name in JPEGImages_names:
        features.append(torchvision.io.read_image(os.path.join(JPEGImages_path, name), mode))
        labels.append(torchvision.io.read_image(os.path.join(Annotations_path, name.split(".")[0] + 
                                                             ".png"), mode))
    return features, labels

def crop_images(feature, label, height = 500, width = 1000):
    # CenterCrop 会自动补0如果图片太小了，所以对feature和label都可以用
    centercrop = torchvision.transforms.CenterCrop((height, width))
    feature = centercrop(feature)
    label = centercrop(label)
    return (feature, label)
