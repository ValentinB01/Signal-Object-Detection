"""
MODEL 1 - CNN simplu (baseline)
Clasificare in 5 clase: numarul de obiecte (1-5) dintr-o spectrograma radio.
Arhitectura: 3 blocuri convolutionale clasice, fara skip connections,
un singur cap de clasificare, CrossEntropyLoss simplu.
Scor Kaggle (public): 0.75272.
Ruleaza end-to-end: antreneaza, salveaza model_run1.pth si genereaza submission_model1.csv.
"""

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

# Split stratificat 80/20, seed fix pentru reproductibilitate
train_df, val_df = train_test_split(
    df_train, test_size=0.2, stratify=df_train['label'], random_state=42
)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)

# Augmentari usoare. Flip-urile sunt valide pentru numarare (numarul de obiecte
# nu se schimba la intoarcere). Normalizarea ImageNet trebuie identica train/test.
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV citeste BGR
        if image.shape[2] == 4:
            image = image[:, :, :3]                     # eliminam canalul alpha daca exista
        label = self.dataframe.loc[idx, 'label'] - 1    # clase 1..5 -> indici 0..4
        if self.transform:
            image = self.transform(image=image)['image']
        return image, torch.tensor(label, dtype=torch.long)

train_dataset = SignalDataset(train_df, TRAIN_IMG_DIR, transform=train_transform)
val_dataset   = SignalDataset(val_df,   TRAIN_IMG_DIR, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  drop_last=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)


class SimpleSignalCNN(nn.Module):
    """CNN clasic cu 3 blocuri (32->64->128). Fiecare bloc: 2x (Conv3x3-BN-ReLU),
    apoi MaxPool si Dropout2d. Fara skip connections, un singur cap de clasificare."""
    def __init__(self, num_classes=5):
        super().__init__()
        # Bloc 1
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.2)
        # Bloc 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.3)
        # Bloc 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.4)
        # Global average pooling -> vector de 128 (robust la dimensiunea intrarii)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Clasificator (un singur cap)
        self.fc1 = nn.Linear(128, 64)
        self.drop_fc = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.drop1(self.pool1(x))
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.drop2(self.pool2(x))
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.drop3(self.pool3(x))
        x = torch.flatten(self.adaptive_pool(x), 1)
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        return self.fc2(x)   # logits pentru cele 5 clase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed fix pentru reproductibilitate
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

model = SimpleSignalCNN(num_classes=5).to(device)
print(f"Model 1 (baseline) pe {device} | Parametri: {sum(p.numel() for p in model.parameters()):,}")

# CrossEntropyLoss simplu (datele sunt echilibrate -> fara class weights)
criterion = nn.CrossEntropyLoss()

EPOCHS = 90
best_val_acc = 0.0

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
# OneCycleLR: warmup 10% pana la max_lr, apoi cosine decay aproape de zero
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=2.5e-3,
    total_steps=EPOCHS * len(train_loader),
    pct_start=0.1, anneal_strategy='cos',
    div_factor=10.0, final_div_factor=1e3
)

for epoch in range(EPOCHS):
    print(f"\n--- Epoca {epoch+1}/{EPOCHS} ---")
    model.train()
    train_correct = train_total = 0
    for images, labels in tqdm(train_loader, desc="Antrenare"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # anti-explozie gradient
        optimizer.step()
        scheduler.step()        # OneCycleLR: pas per-batch
        _, predicted = torch.max(logits, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    epoch_train_acc = train_correct / train_total

    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, predicted = torch.max(logits, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    epoch_val_acc = val_correct / val_total

    print(f"Train Acc: {epoch_train_acc*100:.2f}% | Valid Acc: {epoch_val_acc*100:.2f}%")
    # Salvam cel mai bun model dupa acuratetea de validare
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), 'model_run1.pth')
        print(f"  Record nou - salvat. Top: {best_val_acc*100:.2f}%")
    # Oprire daca modelul colapseaza (prezice o singura clasa)
    if epoch >= 4 and best_val_acc < 0.25:
        print("  Colaps detectat. Oprire.")
        break

print(f"\nAntrenare completa. Best val acc: {best_val_acc*100:.2f}%")

# --- Generarea submisiei cu TTA x4 (4 variante de flip mediate) ---
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
model.load_state_dict(torch.load('model_run1.pth', map_location=device))
model.eval()

predictions = []
with torch.no_grad():
    for images in tqdm(test_loader, desc="Predictie (TTA x4)"):
        images = images.to(device)
        # Mediem probabilitatile pe 4 variante: original + 3 flip-uri
        l1 = model(images)
        l2 = model(torch.flip(images, dims=[3]))     # flip orizontal
        l3 = model(torch.flip(images, dims=[2]))     # flip vertical
        l4 = model(torch.flip(images, dims=[2, 3]))  # ambele
        probs = (torch.softmax(l1, 1) + torch.softmax(l2, 1)
               + torch.softmax(l3, 1) + torch.softmax(l4, 1)) / 4.0
        _, pred = torch.max(probs, 1)
        predictions.extend((pred.cpu().numpy() + 1).tolist())  # indici 0..4 -> clase 1..5

submission_df['label'] = predictions
submission_df.to_csv('submission_model1.csv', index=False)
print("submission_model1.csv generat.")
print(submission_df['label'].value_counts().sort_index())
