import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from model import VoNet  # Import mô hình từ file đã lưu

# Thông số cấu hình
DATASET_DIR = '8_huong_2/test'
BATCH_SIZE = 4
number_of_label = 8  # Số lớp (số góc định nghĩa)
OUTPUT_FEATURES = 2  # Đầu ra là (cos θ, sin θ)
MODEL_PATH = "vonet_trained.pth"  # Đường dẫn tới mô hình đã lưu

# Đường dẫn dữ liệu
TESTDATASET_DIR_IMG = os.path.join(DATASET_DIR, 'images/')
TESTDATASET_DIR_LBL = os.path.join(DATASET_DIR, 'labels/')
IMG_TESTDATASET_PATH_IMG = sorted(glob.glob(f'{TESTDATASET_DIR_IMG}*.jpg'))
IMG_TESTDATASET_PATH_LBL = sorted(glob.glob(f'{TESTDATASET_DIR_LBL}*.txt'))

# Hàm kiểm tra CUDA
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
    transforms.Resize(512, 1024),  # Resize chiều dài lớn nhất về 512, giữ tỷ lệ
    transforms.ToTensor()
])

# Chuẩn bị DataLoader
dataset = DirectionDataset(IMG_TESTDATASET_PATH_IMG, IMG_TESTDATASET_PATH_LBL, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Hàm kiểm tra mô hình
def test_model(model, dataloader):
    model.eval()
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for images, labels, bboxes in dataloader:
            images, bboxes = images.to(device), bboxes.to(device)
            
            # Dự đoán
            preds = model(images, bboxes)
            
            # Lưu dự đoán và nhãn thực tế
            predictions.extend(preds.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())
    
    return predictions, ground_truths

# Tải mô hình đã lưu
model = VoNet()
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
print("Model loaded successfully.")

# Kiểm tra mô hình
print("Starting testing...")
predictions, ground_truths = test_model(model, dataloader)

# Chuyển đổi kết quả từ (cos θ, sin θ) sang góc θ
def convert_to_angle(output):
    x, y = output[:, 0], output[:, 1]
    return np.arctan2(y, x) * (180 / np.pi)  # Chuyển đổi sang độ

pred_angles = convert_to_angle(np.array(predictions))
true_angles = convert_to_angle(np.array(ground_truths))

# Hiển thị một số kết quả
for i in range(5):
    print(f"Sample {i+1}:")
    print(f"  Predicted Angle: {pred_angles[i]:.2f}°")
    print(f"  True Angle: {true_angles[i]:.2f}°")

# Tính toán sai số trung bình
errors = np.abs(pred_angles - true_angles)
mean_error = np.mean(errors)
print(f"Mean Absolute Error (MAE): {mean_error:.2f}°")