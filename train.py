import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from model_torch import LPNet
from model_torch import LPNet_CrackSegmenter

# ==========================================
# 1. 裂缝数据集读取器 (Dataset)
# ==========================================
class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, patch_size=256, is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.patch_size = patch_size
        self.img_names = sorted(os.listdir(img_dir))
        self.mask_names = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = TF.resize(image, (self.patch_size, self.patch_size)) # type: ignore
        mask = TF.resize(mask, (self.patch_size, self.patch_size)) # type: ignore

        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image = TF.to_tensor(image) # type: ignore # shape: [3, 256, 256]
        mask = TF.to_tensor(mask)   # type: ignore # shape: [1, 256, 256]

        mask = (mask > 0.5).float() 

        return image, mask

# ==========================================
# 2. 评价指标计算 (IoU)
# ==========================================
def calculate_iou(pred_logits, true_mask):
    with torch.no_grad():
        # 通过 Sigmoid 并二值化
        pred_mask = (torch.sigmoid(pred_logits) > 0.5).float()
        
        intersection = (pred_mask * true_mask).sum()
        union = pred_mask.sum() + true_mask.sum() - intersection
        
        if union == 0:
            return 1.0 
        return (intersection / union).item()
    
# ==========================================
# 3. 主干训练脚本 (Main)
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. 准备数据 ---
    train_dataset = CrackDataset('train_img', 'train_lab', patch_size=256, is_train=True)
    test_dataset = CrackDataset('test_img', 'test_lab', patch_size=256, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- 2. 准备模型并加载 ---
    base_model = LPNet(num_pyramids=5, num_blocks=5, num_feature=16)
    pretrained_path = 'lpnet.pth'
    if os.path.exists(pretrained_path):
        base_model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
    model = LPNet_CrackSegmenter(base_model).to(device)

    # --- 3. 设置 Loss 和优化器 ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-4}, 
        {'params': model.seg_head.parameters(), 'lr': 1e-3}
    ])

    # --- 4. 开始训练 ---
    num_epochs = 50
    best_iou = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # 训练阶段
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_iou += calculate_iou(logits, masks)
                
        avg_iou = val_iou / len(test_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Test IoU: {avg_iou:.4f}")

        # 保存表现最好的模型
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), 'best_crack_segmenter.pth')
