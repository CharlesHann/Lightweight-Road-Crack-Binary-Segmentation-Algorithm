import cv2
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from model_torch import LPNet
from model_torch import LPNet_CrackSegmenter
import torch
import numpy as np
import matplotlib.pyplot as plt
class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, patch_size=256, is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.patch_size = patch_size
        
        # 假设你的原图和标签文件名是一致的 (比如 001.jpg 和 001.png)
        self.img_names = sorted(os.listdir(img_dir))
        self.mask_names = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])

        # 读取：原图读成 RGB，标签读成灰度图 (L)
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # 统一缩放：因为 LPNet 有 5 层金字塔(下采样16倍)，输入尺寸最好是 16 的倍数
        # 我们统一 Resize 到 256x256 (或者更大，如 512x512)
        image = TF.resize(image, (self.patch_size, self.patch_size)) # type: ignore
        mask = TF.resize(mask, (self.patch_size, self.patch_size)) # type: ignore

        # 训练时加入同步数据增强 (极其重要，防止过拟合！)
        if self.is_train:
            # 50%概率水平翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # 50%概率垂直翻转
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # 转换为 Tensor，值域变成 [0.0, 1.0]
        image = TF.to_tensor(image) # type: ignore # shape: [3, 256, 256]
        mask = TF.to_tensor(mask)   # type: ignore # shape: [1, 256, 256]
        
        mask = (mask > 0.5).float() 

        return image, mask
    
test_dataset = CrackDataset('test_img', 'test_lab', patch_size=256, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
base_model = LPNet(num_pyramids=5, num_blocks=5, num_feature=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LPNet_CrackSegmenter(base_model).to(device)
model.load_state_dict(torch.load('best_crack_segmenter.pth', map_location='cpu'))

def remove_small_blobs(mask_255, min_area=150):
    # 找到所有的独立连通块
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_255, connectivity=8)
    clean_mask = np.zeros_like(mask_255)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == i] = 1.0 
    return clean_mask

def evaluate_pr_curve_with_postprocess(model, test_loader, device, min_area):
    model.eval()
    
    thresholds = np.linspace(0.02, 1, 100)

    TPs = np.zeros(len(thresholds))
    FPs = np.zeros(len(thresholds))
    FNs = np.zeros(len(thresholds))
   
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            # 真实标签 [B, 1, H, W] -> numpy
            true_masks = masks.cpu().numpy().squeeze(1) # shape: [B, H, W]
            
            # 模型推理
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze(1) # shape: [B, H, W]
            
            batch_size = probs.shape[0]
            
            # 对当前 Batch 中的每一张图，遍历所有阈值
            for b in range(batch_size):
                prob_map = probs[b]
                true_map = true_masks[b]
                
                for t_idx, thresh in enumerate(thresholds):
                    # 1. 阈值二值化 (0 和 255)
                    binary_map = (prob_map > thresh).astype(np.uint8) * 255
                    
                    # 2. 连通域过滤后处理
                    # 过滤完之后 clean_map 里只有 0.0 和 1.0
                    clean_map = remove_small_blobs(binary_map, min_area=min_area)
                    
                    # 3. 计算 TP, FP, FN
                    # TP: 预测是1，真实也是1
                    TPs[t_idx] += np.sum((clean_map == 1) & (true_map == 1))
                    # FP: 预测是1，真实是0 (假警报)
                    FPs[t_idx] += np.sum((clean_map == 1) & (true_map == 0))
                    # FN: 预测是0，真实是1 (漏报)
                    FNs[t_idx] += np.sum((clean_map == 0) & (true_map == 1))
                    

    precisions = TPs / (TPs + FPs + 1e-8)
    recalls = TPs / (TPs + FNs + 1e-8)
    f_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # 找到最佳分数
    best_idx = np.argmax(f_scores)
    best_f = f_scores[best_idx]
    best_thresh = thresholds[best_idx]
    
    
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='darkorange', lw=2, 
             label=f'LPNet + CC Filter [Best F={best_f:.3f}]')
    
    plt.scatter(recalls[best_idx], precisions[best_idx], color='red', zorder=5)
    plt.annotate(f'Best F={best_f:.3f}\nThresh={best_thresh:.2f}', 
                 (recalls[best_idx], precisions[best_idx]), 
                 textcoords="offset points", xytext=(-40,-30), ha='center',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('End-to-End Pipeline PR Curve (DeepCrack)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower left")
    plt.savefig('pr_curve_with_postprocess.png', dpi=300)
    print("曲线已保存为 pr_curve_with_postprocess.png")

evaluate_pr_curve_with_postprocess(model, test_loader, device,200)