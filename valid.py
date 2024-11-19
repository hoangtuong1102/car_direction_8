import os
import glob
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Thông số cấu hình
DATASET_DIR = '8_huong/valid'
BATCH_SIZE = 16
number_of_label = 8  # Số lớp (số góc định nghĩa)
OUTPUT_FEATURES = 2  # Đầu ra là (cos θ, sin θ)

# Đường dẫn dữ liệu
VALIDDATASET_DIR_IMG = os.path.join(DATASET_DIR, 'images/')  # Thư mục chứa ảnh
VALIDDATASET_DIR_LBL = os.path.join(DATASET_DIR, 'labels/')  # Thư mục chứa nhãn YOLO
IMG_VALIDDATASET_PATH_IMG = sorted(glob.glob(f'{VALIDDATASET_DIR_IMG}*.jpg'))  # Danh sách ảnh
IMG_VALIDDATASET_PATH_LBL = sorted(glob.glob(f'{VALIDDATASET_DIR_LBL}*.txt'))  # Danh sách nhãn YOLO

# Hàm kiểm tra CUDA
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
            items = lines[0].strip().split()  # Lấy dòng đầu tiên
            class_id = int(items[0])  # Lớp (class_id)
            x_center, y_center, width, height = map(float, items[1:5])
            
            # Chuyển đổi nhãn class_id thành góc (radian)
            angle = np.deg2rad(class_id * (360 / number_of_label))  # Tính góc dựa trên số lớp
            label = [np.cos(angle), np.sin(angle)]  # Đưa góc về dạng (cos θ, sin θ)
            bbox = [x_center, y_center, width, height]  # Lưu bounding box (YOLO format)
        
        return image, torch.tensor(label, dtype=torch.float32), torch.tensor(bbox, dtype=torch.float32)

# Tiền xử lý dữ liệu
transform = transforms.Compose([
    transforms.Resize((2752, 5504)),  # Chuyển đổi tất cả ảnh về cùng kích thước
    transforms.ToTensor()
])

# Chuẩn bị DataLoader
valid_dataset = DirectionDataset(IMG_VALIDDATASET_PATH_IMG, IMG_VALIDDATASET_PATH_LBL, transform)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

# Định nghĩa lại VoNet
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
        )
        self.bbox_fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.25),
            nn.Linear(512 + 8, OUTPUT_FEATURES)  # Kết hợp đặc trưng từ ảnh và bounding box
        )

    def forward(self, x: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        img_features = self.features(x).view(x.size(0), -1)  # [batch_size, 512]
        bbox_features = self.bbox_fc(bbox)  # [batch_size, 8]
        combined_features = torch.cat([img_features, bbox_features], dim=1)  # [batch_size, 520]
        output = self.classifier(combined_features)  # [batch_size, 2]
        return output

# Hàm kiểm tra mô hình
def validate_model(model, dataloader):
    model.to(device)
    model.eval()
    predictions_list = []
    with torch.no_grad():
        for images, labels, bboxes in dataloader:
            images, bboxes = images.to(device), bboxes.to(device)
            predictions = model(images, bboxes)
            predictions_list.append(predictions.cpu().numpy())
    return predictions_list

# Tải mô hình đã lưu
model = VoNet()
model.load_state_dict(torch.load("vonet_trained.pth"))
print("Model loaded successfully.")

# Dự đoán
predictions = validate_model(model, valid_dataloader)
print("Predictions:", predictions[:5])