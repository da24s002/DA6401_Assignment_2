import torch
from torchvision import transforms



######################################## transformation functions #########################
def get_transforms(data_augmentation="No"):
    if data_augmentation=="Yes":
        # Training transforms with data augmentation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5), ## data augmentation
            transforms.RandomVerticalFlip(p=0.3),  ## data augmentation
            transforms.RandomRotation(degrees=15),  ## data augmentation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),  ## data augmentation
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  ## data augmentation
            transforms.ToTensor(),
        ])
    else:
        # Validation/Test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    return transform

## custom transform class for separate transformation behavior of train and validation dataset
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
        
    def __len__(self):
        return len(self.dataset)
    
#######################################################################