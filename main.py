import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from torch.utils.data import DataLoader
from torchvision import transforms
from CityscapesDataset import Cityscapes

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, stride=stride, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResidualDepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualDepthwiseSeparableBlock, self).__init__()
        self.depthwise_separable_conv = DepthwiseSeparableConv(in_channels, out_channels, 3, stride=stride)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.depthwise_separable_conv(x)
        out += self.shortcut(x)
        return F.relu(out)

class SimplifiedUNet(nn.Module):
    def __init__(self, num_classes):
        super(SimplifiedUNet, self).__init__()
        self.in_channels = 64
        self.down1 = ResidualDepthwiseSeparableBlock(3, self.in_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ResidualDepthwiseSeparableBlock(self.in_channels, self.in_channels*2)
        self.pool2 = nn.MaxPool2d(2)

        self.bridge = ResidualDepthwiseSeparableBlock(self.in_channels*2, self.in_channels*4)

        self.up2 = nn.ConvTranspose2d(self.in_channels*4, self.in_channels*2, kernel_size=2, stride=2)
        self.up_conv2 = ResidualDepthwiseSeparableBlock(self.in_channels*4, self.in_channels*2)
        self.up1 = nn.ConvTranspose2d(self.in_channels*2, self.in_channels, kernel_size=2, stride=2)
        self.up_conv1 = ResidualDepthwiseSeparableBlock(self.in_channels*2, self.in_channels)

        self.out = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        p1 = self.pool1(x1)
        x2 = self.down2(p1)
        p2 = self.pool2(x2)

        # Bridge
        bridge = self.bridge(p2)

        # Decoder
        up2 = self.up2(bridge)
        merge2 = torch.cat([up2, x2], dim=1)
        out = self.up_conv2(merge2)
        up1 = self.up1(out)
        merge1 = torch.cat([up1, x1], dim=1)
        out = self.up_conv1(merge1)

        out = self.out(out)
        return out




def get_transform():
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(transform_list)

def transform(image, mask):
    transform_image = get_transform()
    image = transform_image(image)
    
    mask = transforms.ToTensor()(mask)
    return image, mask
train_transform = transform
val_transform = transform

# Define datasets and data loaders
train_dataset = Cityscapes(root_dir=r'/content/drive/MyDrive/disertatie/', split='train', transform=train_transform)
val_dataset = Cityscapes(root_dir=r'/content/drive/MyDrive/disertatie/', split='val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


import torch
import copy
from tqdm import tqdm

from sklearn.metrics import jaccard_score
import numpy as np

def compute_iou_sklearn(preds, labels, num_classes):
    # Convert predictions to binary format and flatten
    preds = torch.argmax(preds, dim=1).cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    
    # Compute IoU. Note that this assumes your labels are in 'binary' form, not one-hot encoded.
    iou = jaccard_score(labels, preds, average='macro', labels=np.arange(num_classes))
    return iou



def train_and_test_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, checkpoint_path='model_checkpoint.pth', checkpoint_interval=1, resume_from=None):
    # Initialize the starting epoch, best IoU score, and copy of the best model weights
    start_epoch = 0
    best_iou = 0.0
    best_acc = 0.0  # Initialize best accuracy

    # If a checkpoint file is provided, load the weights and resume training
    if resume_from is not None:
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
        best_iou = checkpoint['best_iou']
        best_acc = checkpoint['best_acc']  # Resume best accuracy
        print(f"Resuming training from epoch {start_epoch}")

    # Send the model to the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("The model is using the " + str(device) + " device")
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    # Loop over the dataset multiple times
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_iou = 0.0
            running_corrects = 0  # Track the number of correct predictions
            total_samples = 0  # Total number of samples processed
            total_pixels = 0
            correct_pixels = 0

            # Initialize tqdm progress bar
            epoch_progress = tqdm(total=len(train_loader if phase == 'train' else val_loader), desc=f'Epoch {epoch + 1}/{num_epochs} {phase}', unit='batch')

            # Iterate over data
            for inputs, labels in (train_loader if phase == 'train' else val_loader):
                labels = labels.squeeze(1)
                labels = labels.long()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass to get output/logits
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # Get the predictions
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Compute the loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                correct_pixels += (preds == labels).sum().item()
                total_pixels += torch.numel(preds)

                total_samples += inputs.size(0)

                # Compute IoU here (implement your own function or call to a library)
                iou = jaccard_score(labels.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy(), average='macro', labels=np.arange(19), zero_division=1)

                running_iou += iou

                # Dynamically update the progress bar description with the latest loss and accuracy
                epoch_progress.set_description(f'Epoch {epoch + 1}/{num_epochs} {phase} - Loss: {running_loss/total_samples:.4f}, Acc: {correct_pixels/total_pixels:.4f}, IoU: {running_iou/total_samples:.4f}')
                epoch_progress.update(1)

            epoch_progress.close()

            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / total_samples
            epoch_acc = correct_pixels / total_pixels
            epoch_iou = running_iou / total_samples # Calculate epoch IoU

            # Print metrics at the end of the epoch
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} IoU: {epoch_iou:.4f}')

            # Check for best model
            if phase == 'val' and epoch_iou > best_iou:
                best_iou = epoch_iou
                best_acc = epoch_acc  # Update best accuracy
                best_model_wts = copy.deepcopy(model.state_dict())

            # Save checkpoint at specified interval
            if phase == 'train' and (epoch + 1) % checkpoint_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_iou': best_iou,
                    'best_acc': best_acc,  # Save best accuracy
                }, checkpoint_path)

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

model = SimplifiedUNet(19)
criterion = nn.CrossEntropyLoss()  # For multi-class segmentation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_model= train_and_test_model(model,train_loader=train_loader,val_loader=val_loader,criterion=criterion,optimizer=optimizer,checkpoint_path='/content/drive/MyDrive/disertatie/checkpoints')

