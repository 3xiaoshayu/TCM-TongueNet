import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib
matplotlib.use('Agg')
import argparse
import logging
import random
from datetime import datetime
import warnings
import platform

warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        return x_channel * spatial_att

class TongueDataset13Class(Dataset):
    def __init__(self, df, seg_dir, transform=None, preload=False):
        self.df = df.reset_index(drop=True)
        self.seg_dir = seg_dir
        self.transform = transform
        self.preload = preload
        self.label_names = [
            'TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack',
            'Toothmark', 'FurThick', 'FurYellow',
            'Heart', 'Lung', 'Spleen', 'Liver', 'Kidney'
        ]

        if preload:
            print("Preloading images into memory...")
            self.images = []
            for idx in range(len(df)):
                row = df.iloc[idx]
                img_path = os.path.join(seg_dir, row['image_path'])
                try:
                    img = Image.open(img_path).convert('RGB')
                    self.images.append(img)
                except Exception as e:
                    print(f"Warning: Cannot load image {img_path}: {e}")
                    self.images.append(Image.new('RGB', (224, 224), color='black'))
            print(f"Preloading completed, {len(self.images)} images")
        else:
            self.images = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.preload:
            image = self.images[idx]
        else:
            row = self.df.iloc[idx]
            img_path = os.path.join(self.seg_dir, row['image_path'])
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Warning: Cannot load image {img_path}: {e}")
                image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        labels = [self.df.iloc[idx][label] for label in self.label_names]
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels, self.df.iloc[idx]['image_path']

class TongueModel13Class(nn.Module):
    def __init__(self, resnet_version='resnet101', pretrained=True,
                 use_attention=False, attention_type='se', num_classes=13):
        super().__init__()
        import torchvision.models as models
        weights_enum = {
            'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
            'resnet34': models.ResNet34_Weights.IMAGENET1K_V1,
            'resnet50': models.ResNet50_Weights.IMAGENET1K_V1,
            'resnet101': models.ResNet101_Weights.IMAGENET1K_V1,
            'resnet152': models.ResNet152_Weights.IMAGENET1K_V1,
        }
        weights = weights_enum[resnet_version] if pretrained else None
        model_func = getattr(models, resnet_version)
        self.base_model = model_func(weights=weights)
        num_ftrs = self.base_model.fc.in_features

        self.use_attention = use_attention
        self.attention_type = attention_type
        if use_attention:
            if resnet_version in ['resnet18', 'resnet34']:
                channel_sizes = [64, 128, 256, 512]
            else:
                channel_sizes = [256, 512, 1024, 2048]
            Attention = SEAttention if attention_type == 'se' else CBAM
            self.att1 = Attention(channel_sizes[0])
            self.att2 = Attention(channel_sizes[1])
            self.att3 = Attention(channel_sizes[2])
            self.att4 = Attention(channel_sizes[3])

        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        if self.use_attention:
            x = self.att1(x)

        x = self.base_model.layer2(x)
        if self.use_attention:
            x = self.att2(x)

        x = self.base_model.layer3(x)
        if self.use_attention:
            x = self.att3(x)

        x = self.base_model.layer4(x)
        if self.use_attention:
            x = self.att4(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        return x

class WeightedBCELoss(nn.Module):
    def __init__(self, class_weights, device='cuda'):
        super().__init__()
        self.class_weights = class_weights
        self.device = device
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        loss = self.bce(inputs, targets)
        weights = torch.zeros_like(loss)
        for i in range(loss.size(1)):
            neg_mask = (targets[:, i] == 0).float()
            pos_mask = (targets[:, i] == 1).float()
            weights[:, i] = (
                neg_mask * self.class_weights[i][0].to(self.device) +
                pos_mask * self.class_weights[i][1].to(self.device)
            )
        return (loss * weights).mean()

class FocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2, reduction='mean', device='cuda'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.device = device
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        if self.class_weights is not None:
            weights = torch.zeros_like(bce_loss)
            for i in range(bce_loss.size(1)):
                neg_mask = (targets[:, i] == 0).float()
                pos_mask = (targets[:, i] == 1).float()
                weights[:, i] = (
                    neg_mask * self.class_weights[i][0].to(self.device) +
                    pos_mask * self.class_weights[i][1].to(self.device)
                )
            focal_loss = focal_loss * weights
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        self.label_names = [
            'TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack',
            'Toothmark', 'FurThick', 'FurYellow',
            'Heart', 'Lung', 'Spleen', 'Liver', 'Kidney'
        ]
        self.num_classes = len(self.label_names)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        parent_dir = os.path.dirname(args.data_dir)
        self.output_dir = os.path.join(parent_dir, f'13class_output_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        self.logs_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)

        self.setup_logging()
        self.setup_data()
        self.setup_model()

    def setup_logging(self):
        log_file = os.path.join(self.logs_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mode: {self.args.mode}, Optim: {self.args.optim}, simple_aug: {self.args.simple_aug}, loss: {self.args.loss}")

    def setup_data(self):
        img_size = 128 if self.args.mode == 'fast' else 224

        if self.args.simple_aug:
            self.logger.info("Using simplified data augmentation (no rotation/color jitter)")
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.logger.info("Using full data augmentation (with rotation/color jitter)")
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        seg_dir = os.path.join(self.args.data_dir, 'seg')
        list_dir = os.path.join(self.args.data_dir, 'list')

        if os.path.exists(os.path.join(list_dir, 'train.csv')):
            self.logger.info("Using standard CSV split from list/")
            train_csv = os.path.join(list_dir, 'train.csv')
            val_csv = os.path.join(list_dir, 'val.csv')
            test_csv = os.path.join(list_dir, 'test.csv')
        elif os.path.exists(os.path.join(list_dir, 'train_fold1.csv')):
            self.logger.info("Using train_fold1.csv etc. from list/ (legacy)")
            train_csv = os.path.join(list_dir, 'train_fold1.csv')
            val_csv = os.path.join(list_dir, 'val_fold1.csv')
            test_csv = os.path.join(list_dir, 'test.csv')
        else:
            raise FileNotFoundError(f"Training/validation/test CSV files not found in {list_dir}")

        self.train_df = pd.read_csv(train_csv)
        self.val_df = pd.read_csv(val_csv)
        self.test_df = pd.read_csv(test_csv)

        if self.args.mode == 'fast':
            self.train_df = self.train_df.sample(frac=0.05, random_state=42).reset_index(drop=True)
            self.logger.info(f"Fast mode: using {len(self.train_df)} training samples")
        elif self.args.mode == 'medium':
            self.train_df = self.train_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
            self.logger.info(f"Medium mode: using {len(self.train_df)} training samples")

        preload = self.args.preload if hasattr(self.args, 'preload') else False

        self.train_dataset = TongueDataset13Class(self.train_df, seg_dir, self.train_transform, preload=preload)
        self.val_dataset = TongueDataset13Class(self.val_df, seg_dir, self.test_transform, preload=preload)
        self.test_dataset = TongueDataset13Class(self.test_df, seg_dir, self.test_transform, preload=preload)

        batch_map = {'fast': 32, 'medium': 16, 'full': 8}
        self.batch_size = self.args.batch_size if self.args.batch_size else batch_map[self.args.mode]

        if self.args.num_workers is not None:
            num_workers = self.args.num_workers
        else:
            num_workers = 0 if platform.system() == "Windows" else 8

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True)

        self.logger.info(f"Training: {len(self.train_dataset)} | Validation: {len(self.val_dataset)} | Test: {len(self.test_dataset)}")
        self.logger.info(f"Batch size: {self.batch_size}, num_workers: {num_workers}, preload: {preload}")

        self.calculate_class_weights()

    def calculate_class_weights(self):
        pos_counts = [self.train_df[label].sum() for label in self.label_names]
        neg_counts = [len(self.train_df) - pos for pos in pos_counts]
        self.class_weights = []
        for i, (pos, neg) in enumerate(zip(pos_counts, neg_counts)):
            if pos == 0 or neg == 0:
                weights = [1.0, 1.0]
            else:
                pos_ratio = pos / (pos + neg)
                neg_ratio = neg / (pos + neg)
                if i == 6 or i == 10:
                    w_neg, w_pos = 40.0, 1.0
                    total = w_neg + w_pos
                    weights = [w_neg * 2 / total, w_pos * 2 / total]
                else:
                    weights = [pos_ratio * 2, neg_ratio * 2]
            self.class_weights.append(torch.tensor(weights, dtype=torch.float32))
        self.logger.info("Class weights (negative weight, positive weight):")
        for i, name in enumerate(self.label_names):
            self.logger.info(f"  {name}: {self.class_weights[i][0]:.2f}, {self.class_weights[i][1]:.2f}")

    def setup_model(self):
        resnet_map = {'fast': 'resnet18', 'medium': 'resnet50', 'full': 'resnet101'}
        resnet_version = resnet_map[self.args.mode]

        use_attention = 'attention' in self.args.optim
        attention_type = 'se' if 'se' in self.args.optim else 'cbam' if 'cbam' in self.args.optim else None

        self.model = TongueModel13Class(
            resnet_version=resnet_version,
            pretrained=True,
            use_attention=use_attention,
            attention_type=attention_type,
            num_classes=self.num_classes
        ).to(self.device)

        if torch.cuda.device_count() > 1 and not self.args.cpu:
            self.model = nn.DataParallel(self.model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")

        self.model_config = {
            'resnet_version': resnet_version,
            'use_attention': use_attention,
            'attention_type': attention_type,
            'num_classes': self.num_classes
        }

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model: {resnet_version}, attention: {use_attention}({attention_type}), params: {total_params:,}")

        if self.args.loss == 'focal':
            self.criterion = FocalLoss(class_weights=self.class_weights, gamma=self.args.gamma, device=self.device)
            self.logger.info(f"Using Focal Loss (gamma={self.args.gamma})")
        else:
            self.criterion = WeightedBCELoss(self.class_weights, device=self.device)
            self.logger.info("Using Weighted BCE Loss")

        lr_map = {'fast': 1e-3, 'medium': 5e-4, 'full': 1e-4}
        lr = self.args.lr if self.args.lr else lr_map[self.args.mode]
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)

        if self.args.mode == 'fast':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
            )

        self.scaler = torch.cuda.amp.GradScaler() if not self.args.cpu else None

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'val_mean_auc': [], 'val_organ_auc': [], 'val_tongue_auc': [],
            'val_auc_scores': [], 'lr': []
        }

    def compute_metrics(self, outputs, labels):
        probs = torch.sigmoid(outputs.float()).cpu().numpy()
        preds = (probs >= 0.5).astype(np.int32)
        labels_np = labels.cpu().numpy().astype(np.int32)
        acc = (preds == labels_np).mean()
        f1 = f1_score(labels_np, preds, average='macro', zero_division=0)
        return acc, f1

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        for i, (images, labels, _) in enumerate(self.train_loader):
            if epoch == 1 and i == 0:
                print(f"Model device: {next(self.model.parameters()).device}")
                print(f"Images device before to(): {images.device}")
                print(f"Labels device before to(): {labels.device}")

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if epoch == 1 and i == 0:
                print(f"Images device after to(): {images.device}")
                print(f"Labels device after to(): {labels.device}")

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            running_loss += loss.item()

            probs = torch.sigmoid(outputs.float()).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(np.int32)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        train_acc = (all_preds == all_labels).mean()
        train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return running_loss / len(self.train_loader), train_acc, train_f1

    def validate(self, loader):
        self.model.eval()
        val_loss = 0.0
        all_probs = []
        all_preds_bin = []
        all_labels = []
        with torch.no_grad():
            for images, labels, _ in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs.float()).cpu().numpy()
                preds_bin = (probs >= 0.5).astype(np.int32)
                all_probs.append(probs)
                all_preds_bin.append(preds_bin)
                all_labels.append(labels.cpu().numpy())

        all_probs = np.vstack(all_probs)
        all_preds_bin = np.vstack(all_preds_bin)
        all_labels = np.vstack(all_labels)

        acc = (all_preds_bin == all_labels).mean()
        f1 = f1_score(all_labels, all_preds_bin, average='macro', zero_division=0)

        auc_scores = []
        for i in range(self.num_classes):
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            else:
                auc = 0.5
            auc_scores.append(auc)

        mean_auc = np.mean(auc_scores)
        organ_auc = np.mean(auc_scores[8:13])
        tongue_auc = np.mean(auc_scores[0:8])

        return val_loss/len(loader), acc, f1, mean_auc, organ_auc, tongue_auc, auc_scores

    def train(self):
        patience = 30 if self.args.mode == 'full' else (15 if self.args.mode == 'medium' else 5)
        epochs = 150 if self.args.mode == 'full' else (30 if self.args.mode == 'medium' else 15)
        best_auc = 0.0
        patience_counter = 0
        best_epoch = 0

        self.logger.info(f"Starting training, max epochs: {epochs}, early stopping patience: {patience}")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            val_loss, val_acc, val_f1, mean_auc, organ_auc, tongue_auc, auc_scores = self.validate(self.val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['val_mean_auc'].append(mean_auc)
            self.history['val_organ_auc'].append(organ_auc)
            self.history['val_tongue_auc'].append(tongue_auc)
            self.history['val_auc_scores'].append(auc_scores)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            label_auc_str = '  '.join([f"{name}:{auc:.4f}" for name, auc in zip(self.label_names, auc_scores)])
            self.logger.info(f"Epoch {epoch}: train loss={train_loss:.4f}, val loss={val_loss:.4f}")
            self.logger.info(f"  train acc={train_acc:.4f}, train F1={train_f1:.4f}")
            self.logger.info(f"  val acc={val_acc:.4f}, val F1={val_f1:.4f}")
            self.logger.info(f"  val AUC: mean={mean_auc:.4f}, organ={organ_auc:.4f}, tongue={tongue_auc:.4f}")
            self.logger.info(f"  13-class AUC: {label_auc_str}")
            self.logger.info(f"  lr: {self.optimizer.param_groups[0]['lr']:.6f}")

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            if mean_auc > best_auc:
                best_auc = mean_auc
                patience_counter = 0
                best_epoch = epoch
                self.save_checkpoint(epoch, best_auc, 'best')
                self.logger.info(f"  -> New best model (AUC={best_auc:.4f}) saved")
            else:
                patience_counter += 1
                self.logger.info(f"  No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered, stopping training")
                    break

        self.logger.info(f"Training finished, best validation mean AUC: {best_auc:.4f} (epoch {best_epoch})")
        return best_auc

    def save_checkpoint(self, epoch, auc, tag='best'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'best_auc': auc,
            'history': self.history,
            'args': vars(self.args),
            'model_config': self.model_config
        }
        path = os.path.join(self.output_dir, f'13class_{tag}.pth')
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def evaluate_test(self):
        best_path = os.path.join(self.output_dir, '13class_best.pth')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            cfg = checkpoint['model_config']
            self.model = TongueModel13Class(
                resnet_version=cfg['resnet_version'],
                pretrained=False,
                use_attention=cfg['use_attention'],
                attention_type=cfg['attention_type'],
                num_classes=cfg['num_classes']
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded best model (val AUC={checkpoint['best_auc']:.4f})")
        else:
            self.logger.warning("Best model file not found, using current model for evaluation")

        val_loss, val_acc, val_f1, mean_auc, organ_auc, tongue_auc, auc_scores = self.validate(self.test_loader)

        self.logger.info("\n========== Test Set Evaluation Results ==========")
        self.logger.info(f"Mean AUC (13 classes): {mean_auc:.4f}")
        self.logger.info(f"Organ mean AUC (5 classes): {organ_auc:.4f}")
        self.logger.info(f"Tongue mean AUC (8 classes): {tongue_auc:.4f}")
        self.logger.info(f"Test accuracy: {val_acc:.4f}")
        self.logger.info(f"Test macro F1: {val_f1:.4f}")
        for i, name in enumerate(self.label_names):
            self.logger.info(f"  {name}: AUC = {auc_scores[i]:.4f}")

        np.savez(os.path.join(self.output_dir, 'test_predictions.npz'),
                 preds=None, labels=None, auc_scores=auc_scores)

        return mean_auc, organ_auc, tongue_auc, auc_scores

def parse_args():
    parser = argparse.ArgumentParser(description='13-class Tongue Diagnosis Model Training')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Dataset root directory containing seg/ and list/')
    parser.add_argument('--mode', type=str, default='fast',
                        choices=['fast', 'medium', 'full'],
                        help='Training mode: fast (quick validation), medium, full')
    parser.add_argument('--optim', type=str, default='baseline',
                        choices=['baseline', 'attention_se', 'attention_cbam'],
                        help='Optimization scheme: baseline (no attention), attention_se, attention_cbam')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size, default auto-set based on mode (full default 8)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate, default auto-set based on mode')
    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of data loading threads, default 0 on Windows else 8; can be manually set')
    parser.add_argument('--preload', action='store_true', default=False,
                        help='Whether to preload images into memory (default False to avoid memory overflow)')
    parser.add_argument('--simple_aug', action='store_true', default=False,
                        help='Whether to use simplified data augmentation (no rotation/color jitter)')
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss function: bce (weighted cross-entropy), focal (Focal Loss)')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='Gamma parameter for Focal Loss (default 2.0)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    return parser.parse_args()


    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
    trainer.evaluate_test()