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

# Thông số cấu hình
DATASET_DIR = '8_huong_2/train'
BATCH_SIZE = 4
EPOCHS = 50
number_of_label = 8  # Số lớp (số góc định nghĩa)
OUTPUT_FEATURES = 2  # Đầu ra là (cos θ, sin θ)

# Đường dẫn dữ liệu
TESTDATASET_DIR_IMG = os.path.join(DATASET_DIR, 'images/')
TESTDATASET_DIR_LBL = os.path.join(DATASET_DIR, 'labels/')
IMG_TESTDATASET_PATH_IMG = sorted(glob.glob(f'{TESTDATASET_DIR_IMG}*.jpg'))
IMG_TESTDATASET_PATH_LBL = sorted(glob.glob(f'{TESTDATASET_DIR_LBL}*.txt'))

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

# Chuẩn bị DataLoader
dataset = DirectionDataset(IMG_TESTDATASET_PATH_IMG, IMG_TESTDATASET_PATH_LBL, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Hàm huấn luyện
from tqdm import tqdm  # Import thư viện hiển thị tiến trình

# Hàm huấn luyện
def train_model(model, dataloader, epochs=EPOCHS, lr=0.001, save_path="trained_model.pth"):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()
    
    losses = []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0

        # Hiển thị tiến trình batch bằng tqdm
        for batch_idx, (images, labels, bboxes) in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")):
            images, labels, bboxes = images.to(device), labels.to(device), bboxes.to(device)
            
            # Reset gradient
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(images, bboxes)
            
            # Tính loss
            loss = criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            
            # Cập nhật trọng số
            optimizer.step()
            
            # Cộng dồn loss
            epoch_loss += loss.item()
            
            # Hiển thị thông tin batch
            if (batch_idx + 1) % 10 == 0:  # Hiển thị mỗi 10 batch
                print(f"  Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.6f}")

        # Tính loss trung bình cho mỗi epoch
        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.6f}")
    
    # Lưu mô hình sau khi huấn luyện
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    return losses

# Khởi tạo và huấn luyện mô hình
model = VoNet()
losses = train_model(model, dataloader, epochs=EPOCHS, lr=0.0001, save_path="vonet_trained.pth")

# Vẽ đồ thị mất mát
plt.plot(range(len(losses)), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()