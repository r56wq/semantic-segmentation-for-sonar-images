import matplotlib.pyplot as plt
import os
import torchvision

def plot_prediction(model, colormap, val_path, imgname):
    # imgname 不可以有.png 
    img_path = os.join(val_path, "JPEGImages", imgname +".jpg")
    label_path = os.join(val_path, "Annotations", imgname+".png")
    mode = torchvision.io.image.ImageReadMode.RGB
    feature = torchvision.io.read_image(img_path, mode)
    label = torchvision.io.read_image(label_path, mode)
    predicted_classes = model(feature).argmax(dim=0)
    predicted_picture = 
    