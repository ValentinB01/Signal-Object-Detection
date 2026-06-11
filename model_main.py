import os
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Cai catre date (mediu Kaggle)
TRAIN_CSV_PATH  = '/kaggle/input/datasets/buevalentin/sgobjdet/train.csv'
TRAIN_IMG_DIR   = '/kaggle/input/datasets/buevalentin/sgobjdet/train'
TEST_IMG_DIR    = '/kaggle/input/datasets/buevalentin/sgobjdet/test'
SAMPLE_SUB_PATH = '/kaggle/input/datasets/buevalentin/sgobjdet/sample_submission.csv'

df_train = pd.read_csv(TRAIN_CSV_PATH)

# Split stratificat 80/20, acelasi seed ca baseline pentru comparatie corecta
train_df, val_df = train_test_split(
    df_train, test_size=0.2, stratify=df_train['label'], random_state=42
)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)

# Augmentari usoare + normalizare ImageNet (identica train/test)
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

class SignalDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, 'id']
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        label = self.dataframe.loc[idx, 'label'] - 1    # clase 1..5 -> indici 0..4
        if self.transform:
            image = self.transform(image=image)['image']
        return image, torch.tensor(label, dtype=torch.long)

train_dataset = SignalDataset(train_df, TRAIN_IMG_DIR, transform=train_transform)
val_dataset   = SignalDataset(val_df,   TRAIN_IMG_DIR, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  drop_last=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop_p):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch))
        else:
            self.shortcut = nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(drop_p)
    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)          # adunarea skip
        return self.drop(self.pool(out))


class DualHeadSignalCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.block1 = ResidualBlock(3,   32,  drop_p=0.2)
        self.block2 = ResidualBlock(32,  64,  drop_p=0.3)
        self.block3 = ResidualBlock(64,  128, drop_p=0.4)
        self.block4 = ResidualBlock(128, 256, drop_p=0.4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Cap clasificare: 256 -> 128 -> 5
        self.cls_fc1  = nn.Linear(256, 128)
        self.cls_drop = nn.Dropout(0.4)
        self.cls_fc2  = nn.Linear(128, num_classes)
        # Cap regresie: 256 -> 64 -> 1 (fara activare finala = iesire continua)
        self.reg_fc1  = nn.Linear(256, 64)
        self.reg_drop = nn.Dropout(0.3)
        self.reg_fc2  = nn.Linear(64, 1)
    def forward(self, x):
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x)
        x = torch.flatten(self.adaptive_pool(x), 1)   # vector de 256
        c = self.cls_drop(F.relu(self.cls_fc1(x)))
        logits = self.cls_fc2(c)
        r = self.reg_drop(F.relu(self.reg_fc1(x)))
        reg = self.reg_fc2(r).squeeze(1)
        return logits, reg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed fix (diferit de baseline) pentru reproductibilitate
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

model = DualHeadSignalCNN(num_classes=5).to(device)
print(f"Model 2 pe {device} | Parametri: {sum(p.numel() for p in model.parameters()):,}")

# Loss hibrid: clasificare + componenta ordinala (MSE pe valoarea numerica)
criterion_cls = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()
LAMBDA_REG = 0.5   # ponderea regresiei in loss-ul total

EPOCHS = 120
best_val_acc = 0.0

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=2.5e-3,
    total_steps=EPOCHS * len(train_loader),
    pct_start=0.08, anneal_strategy='cos',
    div_factor=10.0, final_div_factor=1e3
)

for epoch in range(EPOCHS):
    print(f"\n--- Epoca {epoch+1}/{EPOCHS} ---")
    model.train()
    train_correct = train_total = 0
    for images, labels in tqdm(train_loader, desc="Antrenare"):
        images, labels = images.to(device), labels.to(device)
        labels_reg = labels.float()                     # tinta pentru capul de regresie
        optimizer.zero_grad()
        logits, reg = model(images)
        loss = criterion_cls(logits, labels) + LAMBDA_REG * criterion_reg(reg, labels_reg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        _, predicted = torch.max(logits, 1)             # predictia vine de la capul de clasificare
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    epoch_train_acc = train_correct / train_total

    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            _, predicted = torch.max(logits, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    epoch_val_acc = val_correct / val_total

    print(f"Train Acc: {epoch_train_acc*100:.2f}% | Valid Acc: {epoch_val_acc*100:.2f}%")
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), 'model_run2.pth')
        print(f"  Record nou - salvat. Top: {best_val_acc*100:.2f}%")
    if epoch >= 4 and best_val_acc < 0.25:
        print("  Colaps detectat. Oprire.")
        break

print(f"\nAntrenare completa. Best val acc: {best_val_acc*100:.2f}%")

print("\nGeneram submisia (TTA x4)...")

class StrictTestDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        img_name = str(self.dataframe.iloc[idx]['id'])
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image

submission_df = pd.read_csv(SAMPLE_SUB_PATH)
test_loader = DataLoader(
    StrictTestDataset(submission_df, TEST_IMG_DIR, transform=val_transform),
    batch_size=32, shuffle=False, num_workers=2)

# Reincarcam cele mai bune greutati salvate
model.load_state_dict(torch.load('model_run2.pth', map_location=device))
model.eval()

predictions = []
with torch.no_grad():
    for images in tqdm(test_loader, desc="Predictie (TTA x4)"):
        images = images.to(device)
        # Folosim DOAR capul de clasificare (ignoram regresia la inferenta)
        l1, _ = model(images)
        l2, _ = model(torch.flip(images, dims=[3]))     # flip orizontal
        l3, _ = model(torch.flip(images, dims=[2]))     # flip vertical
        l4, _ = model(torch.flip(images, dims=[2, 3]))  # ambele
        probs = (torch.softmax(l1, 1) + torch.softmax(l2, 1)
               + torch.softmax(l3, 1) + torch.softmax(l4, 1)) / 4.0
        _, pred = torch.max(probs, 1)
        predictions.extend((pred.cpu().numpy() + 1).tolist())  # indici 0..4 -> clase 1..5

submission_df['label'] = predictions
submission_df.to_csv('submission_model2.csv', index=False)
print("submission_model2.csv generat.")
print(submission_df['label'].value_counts().sort_index())
