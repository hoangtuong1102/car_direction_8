import os
import glob
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch import optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm  # Hiển thị tiến trình

# Thông số cấu hình
TRAIN_DIR = '8_huong_2/train'
VALID_DIR = '8_huong_2/valid'
BATCH_SIZE = 4
EPOCHS = 50
number_of_label = 8  # Số lớp (số góc định nghĩa)
OUTPUT_FEATURES = 2  # Đầu ra là (cos θ, sin θ)

# Hàm kiểm tra CUDA
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Định nghĩa các module IModule và FModule
class IModule(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int):
        super(IModule, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_dim, h_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=1),
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        return torch.cat([branch1, branch2], 1)

class FModule(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(FModule, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear1(x)

# Định nghĩa VoNet với Bounding Box
class VoNet(nn.Module):
    def __init__(self):
        super(VoNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            IModule(in_dim=64, h_dim=96, out_dim=128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FModule(256, 256),
            FModule(256, 384),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FModule(384, 384),
            FModule(384, 512),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.bbox_fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(512 + 8, OUTPUT_FEATURES)  # Kết hợp đặc trưng từ ảnh và bounding box
        )

    def forward(self, x: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        img_features = self.features(x).view(x.size(0), -1)  # [batch_size, 512]
        bbox_features = self.bbox_fc(bbox)  # [batch_size, 8]
        combined_features = torch.cat([img_features, bbox_features], dim=1)  # [batch_size, 520]
        output = self.classifier(combined_features)  # [batch_size, 2]
        return output

# Định nghĩa Dataset
class DirectionDataset(Dataset):
    def __init__(self, img_paths, lbl_paths, transform):
        self.img_paths = img_paths
        self.lbl_paths = lbl_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        lbl_path = self.lbl_paths[idx]
        
        # Xử lý ảnh
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Xử lý nhãn YOLOv11
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                raise ValueError(f"Empty label file: {lbl_path}")
            items = lines[0].strip().split()
            class_id = int(items[0])
            x_center, y_center, width, height = map(float, items[1:5])
            
            # Chuyển đổi nhãn class_id thành góc (radian)
            angle = np.deg2rad(class_id * (360 / number_of_label))
            label = [np.cos(angle), np.sin(angle)]
            bbox = [x_center, y_center, width, height]
        
        return image, torch.tensor(label, dtype=torch.float32), torch.tensor(bbox, dtype=torch.float32)

# Tiền xử lý dữ liệu
transform = transforms.Compose([
    transforms.Resize(512),  # Resize chiều dài lớn nhất về 512, giữ tỷ lệ
    transforms.ToTensor()
])

# Dữ liệu train và valid
train_img_paths = sorted(glob.glob(f'{TRAIN_DIR}/images/*.jpg'))
train_lbl_paths = sorted(glob.glob(f'{TRAIN_DIR}/labels/*.txt'))
valid_img_paths = sorted(glob.glob(f'{VALID_DIR}/images/*.jpg'))
valid_lbl_paths = sorted(glob.glob(f'{VALID_DIR}/labels/*.txt'))

train_dataset = DirectionDataset(train_img_paths, train_lbl_paths, transform)
valid_dataset = DirectionDataset(valid_img_paths, valid_lbl_paths, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Hàm đánh giá
def validate_model(model, dataloader):
    model.eval()
    total_loss = 0
    criterion = nn.SmoothL1Loss()
    with torch.no_grad():
        for images, labels, bboxes in dataloader:
            images, labels, bboxes = images.to(device), labels.to(device), bboxes.to(device)
            predictions = model(images, bboxes)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Hàm huấn luyện
def train_model(model, train_loader, valid_loader, epochs=EPOCHS, lr=0.001, save_path="trained_model.pth"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()
    train_losses, valid_losses = [], []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0.0

        for images, labels, bboxes in tqdm(train_loader, desc="Training"):
            images, labels, bboxes = images.to(device), labels.to(device), bboxes.to(device)
            
            optimizer.zero_grad()
            predictions = model(images, bboxes)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Training Loss: {epoch_loss:.6f}")

        # Validation
        valid_loss = validate_model(model, valid_loader)
        valid_losses.append(valid_loss)
        print(f"Validation Loss: {valid_loss:.6f}")
    
    # Lưu mô hình
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return train_losses, valid_losses

# Khởi tạo và huấn luyện mô hình
model = VoNet()
train_losses, valid_losses = train_model(model, train_loader, valid_loader, epochs=EPOCHS, lr=0.0001, save_path="vonet_trained.pth")

# Vẽ đồ thị mất mát
plt.plot(range(EPOCHS), train_losses, label="Training Loss")
plt.plot(range(EPOCHS), valid_losses, label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('Training_and_Validation_Loss.png')
plt.show()