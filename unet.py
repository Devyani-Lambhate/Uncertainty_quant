import os
import torch
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import random_split
from lidar_to_bev import lidar_to_bev

# Initialize nuScenes object, modify the dataroot and version as per your setup
NUSCENES_DATAROOT = '/home/saksham/samsad/mtech-project/datasets/nuscenes/'
NUSCENES_VERSION = 'v1.0-mini'  # or 'v1.0-trainval' etc.
#nuscenes = NuScenes(NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)


# Define a custom Dataset loading images and dummy regression targets
class NuScenesRegressionDataset(Dataset):

    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

        # Load only image files (filter out directories)
        all_files = sorted(os.listdir(input_dir))
        self.images = [f for f in all_files if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.images[idx])
        target_path = os.path.join(self.target_dir, self.images[idx])

        
        #input_points = np.fromfile(input_path, dtype=np.float32).reshape(-1, 5)[:, :4]  # x, y, z, intensity
        #target_points = np.fromfile(target_path, dtype=np.float32).reshape(-1, 5)[:, :4]  # x, y, z, intensity
        #load images
        input_image = Image.open(input_path)
        target_image = Image.open(target_path)

        #input = lidar_to_bev(input_points)
        #target = lidar_to_bev(target_points)

        #input_image = Image.fromarray(input_points)
        #target_image = Image.fromarray(target_points)
        #target_image = Image.fromarray((target[:,:,1]-input[:,:,1])**2)

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image
    

# Define transforms: resize, to tensor, normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats for RGB
                         std=[0.229, 0.224, 0.225]),
])

# Create dataset and dataloader
train_dataset = NuScenesRegressionDataset(target_dir='/home/user/Documents/dev_exps/sensor-fusion/OpenPCDet/data/nuscenes/v1.0-mini/samples/CAM_FRONT_ORG', input_dir='/home/user/Documents/dev_exps/nuscenes-weather-camera/samples/fog/fog_alpha_0.04/CAM_FRONT',transform=transform)
test_dataset = NuScenesRegressionDataset(target_dir='/home/user/Documents/dev_exps/sensor-fusion/OpenPCDet/data/nuscenes/v1.0-mini/samples/CAM_FRONT_ORG', input_dir='/home/user/Documents/dev_exps/nuscenes-weather-camera/samples/fog/fog_alpha_0.04/CAM_FRONT',transform=transform)
total_size = len(train_dataset)
print("Total dataset size:", total_size)
#train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

# Modify UNet to accept 3 channel input instead of 1
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, 3, padding=1),
                                  nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2)    
        
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, 3, padding=1),
                                  nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 256, 3, padding=1),
                                  nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(512, 512, 3, padding=1),
                                  nn.ReLU(inplace=True))
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(1024, 1024, 3, padding=1),
                                        nn.ReLU(inplace=True))
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = nn.Sequential(nn.Conv2d(1024, 512, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(512, 512, 3, padding=1),
                                  nn.ReLU(inplace=True))
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 256, 3, padding=1),
                                  nn.ReLU(inplace=True))
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, 3, padding=1),
                                  nn.ReLU(inplace=True))
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, 3, padding=1),
                                  nn.ReLU(inplace=True))
        
        # Output layer
        self.conv_last = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        u4 = self.upconv4(b)
        u4 = torch.cat((u4, e4), dim=1)
        d4 = self.dec4(u4)
        
        u3 = self.upconv3(d4)
        u3 = torch.cat((u3, e3), dim=1)
        d3 = self.dec3(u3)
        
        u2 = self.upconv2(d3)
        u2 = torch.cat((u2, e2), dim=1)
        d2 = self.dec2(u2)
        
        u1 = self.upconv1(d2)
        u1 = torch.cat((u1, e1), dim=1)
        d1 = self.dec1(u1)
        
        out = self.conv_last(d1)
        return out
    
import matplotlib.pyplot as plt

def unnormalize(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.copy()
    img = img * std[:, None, None] + mean[:, None, None]  # scale and shift
    img = np.clip(img, 0, 1)  # clip to valid range
    return img

def clip_for_plot(img):
        """Clip image values between foreground min/max, ignoring background (near 0/255)."""
        img=img*255
        foreground = img[(img > 10) & (img < 255)]
        if len(foreground) > 0:
            vmin, vmax = np.min(foreground), np.max(foreground)
            return np.clip(img, vmin, vmax)
        return img

def predict_and_plot(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    inputs, targets = next(iter(dataloader))  # get a batch from test data
    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        outputs = model(inputs)
    
    # Move tensors to CPU and convert to numpy for plotting
    inputs_np = inputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    outputs_np = outputs.cpu().numpy()

    batch_size = inputs_np.shape[0]
    
    

    for i in range(min(batch_size, 4)):  # Plot first 4 samples
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        # Input image (transpose CHW to HWC for 3 channels)
        print(np.min(inputs_np[i]),np.max(inputs_np[i]))
        print(np.min(targets_np[i]),np.max(targets_np[i]))
        input_img = np.transpose(inputs_np[i], (1, 2, 0))  # Convert from CHW to HWC
        target_img = np.transpose(targets_np[i], (1, 2, 0))
        output_img = np.transpose(outputs_np[i], (1, 2, 0))
        
        # Ground truth regression target (3 channels)
        ax[0].imshow(np.abs(input_img - target_img))
        ax[0].set_title("|Simulated weather - Clear weather|")
        ax[0].axis("off")
        
        # Predicted regression output (3 channels)
        ax[1].imshow(np.abs(input_img - output_img))
        ax[1].set_title("|Simulated weather - Predicted Clear weather")
        ax[1].axis("off")
        
        #plt.show()
        plt.savefig("prediction_{}.png".format(i))


# Training loop (same as before)
def train_unet_regression():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=3).to(device)
    
    criterion = torch.nn.MSELoss()
    #print(list(model.parameters()))  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 5
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in train_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            
        epoch_loss /= len(train_dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "unet_regression_model.pth")
    print("Model saved to unet_regression_model.pth")

    predict_and_plot(model, test_dataloader)

if __name__ == "__main__":
    train_unet_regression()
