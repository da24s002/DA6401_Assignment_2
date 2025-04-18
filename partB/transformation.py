
from torchvision import transforms

############################################## Define transformations for the training data and create dataset and dataloader  ################################

"""
Here the first parameter, transformas.Resize(256), resizes the input image dimension to 256x256x3.

transforms.CenterCrop(224), takes a 224x224 patch from it, making the final dimension 224x224x3.

transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) is used, since, 
these specific mean ([0.485, 0.456, 0.406]) and standard deviation ([0.229, 0.224, 0.225]) values are 
the channel-wise (RGB) statistics calculated from the ImageNet dataset, so we are converting our image to match them for better results.

"""

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
