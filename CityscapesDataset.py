import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class Cityscapes(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.images_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.masks_dir = os.path.join(root_dir, 'gtFine', split)
        
        self.images = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(self.images_dir) for f in filenames if f.endswith('.png')])
        self.masks = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(self.masks_dir) for f in filenames if f.endswith('_labelIds.png')])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask