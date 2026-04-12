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
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import random
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

_set_seed_called = False


def set_seed(seed=42, verbose=True):
    global _set_seed_called

    if _set_seed_called and not verbose:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if verbose and not _set_seed_called:
        print(f"Random seed set to: {seed}")
        _set_seed_called = True


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
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
        x_spatial = x_channel * spatial_att
        return x_spatial


class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()
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


class TongueDataset5Class(Dataset):
    def __init__(self, df, seg_dir, transform=None, label_names=None):
        self.df = df
        self.seg_dir = seg_dir
        self.transform = transform
        self.label_names = label_names or ['Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.seg_dir, row['image_path'])

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Cannot load image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        labels = []
        for label in self.label_names:
            labels.append(row[label])
        labels = torch.tensor(labels, dtype=torch.float32)

        img_name = row['image_path']
        return image, labels, img_name


class TongueModel5Class(nn.Module):
    def __init__(self, resnet_version='resnet101', pretrained=True, use_attention=None, attention_type='cbam'):
        super(TongueModel5Class, self).__init__()

        if pretrained:
            import torchvision.models as models
            if resnet_version == 'resnet18':
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            elif resnet_version == 'resnet34':
                weights = models.ResNet34_Weights.IMAGENET1K_V1
            elif resnet_version == 'resnet50':
                weights = models.ResNet50_Weights.IMAGENET1K_V1
            elif resnet_version == 'resnet101':
                weights = models.ResNet101_Weights.IMAGENET1K_V1
            elif resnet_version == 'resnet152':
                weights = models.ResNet152_Weights.IMAGENET1K_V1
            else:
                raise ValueError(f"Unsupported ResNet version: {resnet_version}")
        else:
            weights = None

        import torchvision.models as models
        if resnet_version == 'resnet18':
            self.base_model = models.resnet18(weights=weights)
            num_ftrs = self.base_model.fc.in_features
        elif resnet_version == 'resnet34':
            self.base_model = models.resnet34(weights=weights)
            num_ftrs = self.base_model.fc.in_features
        elif resnet_version == 'resnet50':
            self.base_model = models.resnet50(weights=weights)
            num_ftrs = self.base_model.fc.in_features
        elif resnet_version == 'resnet101':
            self.base_model = models.resnet101(weights=weights)
            num_ftrs = self.base_model.fc.in_features
        elif resnet_version == 'resnet152':
            self.base_model = models.resnet152(weights=weights)
            num_ftrs = self.base_model.fc.in_features
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}")

        self.use_attention = use_attention
        self.attention_type = attention_type

        if use_attention:
            if attention_type == 'cbam':
                self.attention1 = CBAM(64) if resnet_version in ['resnet18', 'resnet34'] else CBAM(256)
                self.attention2 = CBAM(128) if resnet_version in ['resnet18', 'resnet34'] else CBAM(512)
                self.attention3 = CBAM(256) if resnet_version in ['resnet18', 'resnet34'] else CBAM(1024)
                self.attention4 = CBAM(512) if resnet_version in ['resnet18', 'resnet34'] else CBAM(2048)
            elif attention_type == 'se':
                self.attention1 = SEAttention(64) if resnet_version in ['resnet18', 'resnet34'] else SEAttention(256)
                self.attention2 = SEAttention(128) if resnet_version in ['resnet18', 'resnet34'] else SEAttention(512)
                self.attention3 = SEAttention(256) if resnet_version in ['resnet18', 'resnet34'] else SEAttention(1024)
                self.attention4 = SEAttention(512) if resnet_version in ['resnet18', 'resnet34'] else SEAttention(2048)

        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5)
        )

        self.label_names = ['Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        if self.use_attention:
            x = self.attention1(x)

        x = self.base_model.layer2(x)
        if self.use_attention:
            x = self.attention2(x)

        x = self.base_model.layer3(x)
        if self.use_attention:
            x = self.attention3(x)

        x = self.base_model.layer4(x)
        if self.use_attention:
            x = self.attention4(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        return x


class WeightedBCELoss(nn.Module):
    def __init__(self, class_weights, device='cuda'):
        super(WeightedBCELoss, self).__init__()
        self.class_weights = class_weights
        self.device = device
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        num_classes = inputs.size(1)
        loss = self.bce(inputs, targets)
        weights = torch.zeros_like(loss)

        for i in range(num_classes):
            neg_mask = (targets[:, i] == 0).float()
            pos_mask = (targets[:, i] == 1).float()
            weights[:, i] = (
                    neg_mask * self.class_weights[i][0].to(self.device) +
                    pos_mask * self.class_weights[i][1].to(self.device)
            )

        weighted_loss = loss * weights
        return weighted_loss.mean()


class TongueModelTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_dir_parent = os.path.dirname(self.args.data_dir)
        self.optimized_output_dir = os.path.join(data_dir_parent, f'optimized_output_{timestamp}')
        os.makedirs(self.optimized_output_dir, exist_ok=True)

        self.logs_dir = os.path.join(self.optimized_output_dir, 'logs_optimized')
        os.makedirs(self.logs_dir, exist_ok=True)

        self.setup_logging()

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_auc_scores': [],
            'test_auc': [],
            'test_auc_scores': [],
            'lr': []
        }

        self.best_model_state = None
        self.best_val_auc = 0.0
        self.best_epoch = 0

        self.setup_data()
        self.setup_model()

    def setup_logging(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.logs_dir, f'training_{timestamp}.log')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Optimized output directory: {self.optimized_output_dir}")
        self.logger.info(f"Debug mode: {self.args.mode}")
        self.logger.info(f"Optimization scheme: {self.args.optim}")
        self.logger.info(f"Stability optimizations: random seed set, adjusted early stopping patience")

    def setup_data(self):
        self.logger.info("Setting up datasets...")

        if self.args.mode == 'fast':
            img_size = 128
        else:
            img_size = 224

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
        new_list_dir = os.path.join(self.args.data_dir, 'list_new')

        if os.path.exists(new_list_dir):
            self.logger.info("Using new dataset split (list_new/)")
            train_csv = os.path.join(new_list_dir, 'train.csv')
            val_csv = os.path.join(new_list_dir, 'val.csv')
            test_csv = os.path.join(new_list_dir, 'test.csv')

            self.train_df = pd.read_csv(train_csv)
            self.val_df = pd.read_csv(val_csv)
            self.test_df = pd.read_csv(test_csv)

            self.logger.info(f"  Training set: {len(self.train_df)} samples")
            self.logger.info(f"  Validation set: {len(self.val_df)} samples")
            self.logger.info(f"  Test set: {len(self.test_df)} samples")
        else:
            self.logger.info("Using original dataset split")
            train_csv = os.path.join(self.args.data_dir, 'list', 'train_fold1.csv')
            val_csv = os.path.join(self.args.data_dir, 'list', 'val_fold1.csv')
            test_csv = os.path.join(self.args.data_dir, 'list', 'test.csv')

            self.train_df = pd.read_csv(train_csv)
            self.val_df = pd.read_csv(val_csv)
            self.test_df = pd.read_csv(test_csv)

        if self.args.mode == 'fast':
            subset_size = int(len(self.train_df) * 0.05)
            self.train_df = self.train_df.sample(n=subset_size, random_state=42).reset_index(drop=True)
            self.logger.info(f"Fast mode: using {subset_size} training samples (5%)")
        elif self.args.mode == 'medium':
            subset_size = int(len(self.train_df) * 0.2)
            self.train_df = self.train_df.sample(n=subset_size, random_state=42).reset_index(drop=True)
            self.logger.info(f"Medium mode: using {subset_size} training samples (20%)")

        self.train_dataset = TongueDataset5Class(self.train_df, seg_dir, transform=self.train_transform)
        self.val_dataset = TongueDataset5Class(self.val_df, seg_dir, transform=self.test_transform)
        self.test_dataset = TongueDataset5Class(self.test_df, seg_dir, transform=self.test_transform)

        self.calculate_class_weights_balanced()
        self.create_data_loaders()

        self.logger.info(f"Training set size: {len(self.train_dataset)} samples")
        self.logger.info(f"Validation set size: {len(self.val_dataset)} samples")
        self.logger.info(f"Test set size: {len(self.test_dataset)} samples")

    def calculate_class_weights_balanced(self):
        label_columns = ['Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']
        pos_counts = []
        neg_counts = []

        for label in label_columns:
            pos_counts.append(self.train_df[label].sum())
            neg_counts.append(len(self.train_df) - pos_counts[-1])

        self.class_weights = []
        self.logger.info("Class sample statistics:")
        categories = ['Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']
        for i, cat in enumerate(categories):
            self.logger.info(f"  {cat}: {pos_counts[i]} positive, {neg_counts[i]} negative")

        self.logger.info("\nComputing class weights - Spleen using 40:1 weight (best configuration):")

        for i, (pos, neg) in enumerate(zip(pos_counts, neg_counts)):
            if pos > 0 and neg > 0:
                pos_ratio = pos / (pos + neg)
                neg_ratio = neg / (pos + neg)

                if i == 2:
                    weight_for_neg = 40.0
                    weight_for_pos = 1.0
                    total = weight_for_neg + weight_for_pos
                    weights = torch.tensor([
                        weight_for_neg * 2 / total,
                        weight_for_pos * 2 / total
                    ], dtype=torch.float32)
                    actual_ratio = weights[0] / weights[1]
                    self.logger.info(f"  Spleen weight correction: negative weight={weights[0]:.2f}, positive weight={weights[1]:.2f} (ratio {actual_ratio:.1f}:1)")
                else:
                    weights = torch.tensor([neg_ratio * 2, pos_ratio * 2], dtype=torch.float32)
            else:
                weights = torch.tensor([1.0, 1.0], dtype=torch.float32)

            self.class_weights.append(weights)

        self.logger.info("\nFinal class weights (Spleen 40:1):")
        for i, cat in enumerate(categories):
            weight_str = f"negative: {self.class_weights[i][0]:.2f}, positive: {self.class_weights[i][1]:.2f}"
            self.logger.info(f"  {cat}: {weight_str}")

    def create_data_loaders(self):
        num_workers = 8
        if self.args.batch_size is not None:
            effective_batch_size = self.args.batch_size
        elif self.args.mode == 'fast':
            effective_batch_size = 32
        elif self.args.mode == 'medium':
            effective_batch_size = 16
        else:
            effective_batch_size = 12

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return effective_batch_size

    def setup_model(self):
        self.logger.info("Initializing model...")

        if self.args.mode == 'fast':
            model_name = 'resnet18'
        elif self.args.mode == 'medium':
            model_name = 'resnet50'
        else:
            model_name = 'resnet101'

        use_attention = False
        attention_type = 'cbam'
        if 'attention' in self.args.optim:
            use_attention = True
            if 'se' in self.args.optim:
                attention_type = 'se'

        self.model = TongueModel5Class(
            resnet_version=model_name,
            pretrained=True,
            use_attention=use_attention,
            attention_type=attention_type
        ).to(self.device)

        if torch.cuda.device_count() > 1 and not self.args.cpu:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Use attention: {use_attention} ({attention_type if use_attention else 'none'})")
        self.logger.info(f"Trainable parameters: {total_params:,}")

        if self.class_weights is not None:
            self.criterion = WeightedBCELoss(self.class_weights, device=self.device)
            self.logger.info("Using weighted BCEWithLogitsLoss (Spleen 40:1 weight)")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            self.logger.info("Using BCEWithLogitsLoss")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr,
                                     weight_decay=self.args.weight_decay * 2)

        if self.args.mode == 'fast':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5,
                patience=10, min_lr=1e-7
            )
            self.logger.info("Using ReduceLROnPlateau learning rate scheduler (patience 10 epochs)")

    def train_epoch(self, epoch, use_mixup=False):
        self.model.train()
        running_loss = 0.0
        total_batches = len(self.train_loader)
        epoch_start_time = time.time()

        for i, (images, labels, _) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 20 == 0:
                progress = (i + 1) / total_batches * 100
                elapsed = time.time() - epoch_start_time
                batches_per_sec = (i + 1) / elapsed
                self.logger.info(f'   [Epoch {epoch}, Batch {i + 1}/{total_batches}] '
                                 f'Loss: {loss.item():.4f}, Progress: {progress:.1f}%, '
                                 f'Speed: {batches_per_sec:.1f} batches/sec')

        epoch_time = time.time() - epoch_start_time
        self.logger.info(f'[Performance] Epoch {epoch} completed: {epoch_time:.1f} sec, '
                         f'Average speed: {total_batches / epoch_time:.1f} batches/sec')

        return running_loss / total_batches

    def validate(self, loader):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for images, labels, _ in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.sigmoid(outputs.float()).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        all_outputs = np.vstack(all_outputs)

        auc_scores = []
        for i in range(5):
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                auc_scores.append(auc)
            else:
                auc_scores.append(0.5)

        mean_auc = np.mean(auc_scores)
        return val_loss / len(loader), mean_auc, auc_scores, all_preds, all_labels, all_outputs

    def plot_loss_curves(self):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.history['train_loss']) + 1)

        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.optimized_output_dir, 'loss_curves_optimized.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Loss curves saved to: {save_path}")

    def plot_roc_curves(self, all_preds, all_labels):
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        label_names = ['Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']

        for i, label in enumerate(label_names):
            if len(np.unique(all_labels[:, i])) > 1:
                fpr, tpr, _ = roc_curve(all_labels[:, i], all_preds[:, i])
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                plt.plot(fpr, tpr, color=colors[i], lw=2,
                         label=f'{label} (AUC = {auc:.3f})')

        all_labels_flat = all_labels.ravel()
        all_preds_flat = all_preds.ravel()
        if len(np.unique(all_labels_flat)) > 1:
            fpr_micro, tpr_micro, _ = roc_curve(all_labels_flat, all_preds_flat)
            auc_micro = roc_auc_score(all_labels_flat, all_preds_flat)
            plt.plot(fpr_micro, tpr_micro, color='black', linestyle='--', lw=3,
                     label=f'Micro-average (AUC = {auc_micro:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for All Classes', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.optimized_output_dir, 'roc_curves_optimized.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"ROC curves saved to: {save_path}")

    def plot_auc_curves(self):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.history['val_auc_scores']) + 1)
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        label_names = ['Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']

        auc_scores_by_class = list(zip(*self.history['val_auc_scores']))

        for i, label in enumerate(label_names):
            if i < len(auc_scores_by_class):
                plt.plot(epochs, auc_scores_by_class[i], color=colors[i], lw=2,
                         label=f'{label} AUC', marker='o', markersize=3)

        plt.plot(epochs, self.history['val_auc'], color='black', lw=3,
                 label='Mean AUC', linestyle='--', marker='s', markersize=5)

        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('AUC', fontsize=12)
        plt.title('AUC Progression During Training', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim([0.0, 1.05])

        save_path = os.path.join(self.optimized_output_dir, 'auc_progression_optimized.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"AUC progression saved to: {save_path}")

    def plot_class_distribution(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        datasets = [self.train_df, self.val_df, self.test_df]
        dataset_names = ['Training Set', 'Validation Set', 'Test Set']
        label_names = ['Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']

        for idx, (df, name) in enumerate(zip(datasets, dataset_names)):
            counts = []
            for label in label_names:
                count = sum(1 for val in df[label].values if val == 1.0)
                counts.append(count)

            axes[idx].bar(range(len(label_names)), counts, color='skyblue', edgecolor='black')
            axes[idx].set_xlabel('Classes', fontsize=11)
            axes[idx].set_ylabel('Number of Samples', fontsize=11)
            axes[idx].set_title(f'{name} (n={len(df)})', fontsize=12, fontweight='bold')
            axes[idx].set_xticks(range(len(label_names)))
            axes[idx].set_xticklabels(label_names, rotation=45, ha='right')

            for i, count in enumerate(counts):
                axes[idx].text(i, count + max(counts) * 0.01, str(count),
                               ha='center', va='bottom', fontsize=9)

        plt.suptitle('Class Distribution Across Datasets', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()

        save_path = os.path.join(self.optimized_output_dir, 'class_distribution_optimized.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Class distribution saved to: {save_path}")

    def plot_confusion_matrices(self, all_preds, all_labels, threshold=0.5):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        label_names = ['Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']

        for i, label in enumerate(label_names):
            if i < len(axes) - 1:
                y_pred_bin = (all_preds[:, i] >= threshold).astype(int)
                y_true = all_labels[:, i].astype(int)
                cm = confusion_matrix(y_true, y_pred_bin)

                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            cbar=False, square=True, ax=axes[i])
                axes[i].set_xlabel('Predicted Label', fontsize=10)
                axes[i].set_ylabel('True Label', fontsize=10)
                axes[i].set_title(f'{label} Confusion Matrix', fontsize=11, fontweight='bold')
                axes[i].set_xticklabels(['Negative', 'Positive'])
                axes[i].set_yticklabels(['Negative', 'Positive'], rotation=0)

        if len(label_names) < len(axes):
            fig.delaxes(axes[-1])

        plt.suptitle('Confusion Matrices for All Classes (Threshold=0.5)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = os.path.join(self.optimized_output_dir, 'confusion_matrices_optimized.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Confusion matrices saved to: {save_path}")

    def plot_prediction_distributions(self, all_preds, all_labels):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        label_names = ['Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']

        for i, label in enumerate(label_names):
            if i < len(axes) - 1:
                preds_pos = all_preds[all_labels[:, i] == 1, i]
                preds_neg = all_preds[all_labels[:, i] == 0, i]

                axes[i].hist(preds_neg, bins=30, alpha=0.6, color='red',
                             label='Negative', density=True)
                axes[i].hist(preds_pos, bins=30, alpha=0.6, color='blue',
                             label='Positive', density=True)
                axes[i].axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.7)
                axes[i].set_xlabel('Prediction Probability', fontsize=10)
                axes[i].set_ylabel('Density', fontsize=10)
                axes[i].set_title(f'{label} Prediction Distribution', fontsize=11, fontweight='bold')
                axes[i].legend(fontsize=9)
                axes[i].set_xlim([0, 1])

        if len(label_names) < len(axes):
            fig.delaxes(axes[-1])

        plt.suptitle('Prediction Probability Distributions by True Label',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = os.path.join(self.optimized_output_dir, 'prediction_distributions_optimized.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Prediction distributions saved to: {save_path}")

    def generate_all_plots(self, test_preds, test_labels):
        self.logger.info("\nGenerating visualizations...")
        self.plot_loss_curves()
        self.plot_roc_curves(test_preds, test_labels)
        self.plot_auc_curves()
        self.plot_class_distribution()
        self.plot_confusion_matrices(test_preds, test_labels)
        self.plot_prediction_distributions(test_preds, test_labels)
        self.logger.info("All visualizations generated and saved")

    def save_training_summary(self, test_auc, test_auc_scores):
        summary_path = os.path.join(self.optimized_output_dir, 'training_summary_optimized.txt')

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Tongue Diagnosis Model Optimized Training Summary\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Training parameters:\n")
            f.write(f"  Mode: {self.args.mode}\n")
            f.write(f"  Optimization scheme: {self.args.optim}\n")
            f.write(f"  Batch size: {self.args.batch_size}\n")
            f.write(f"  Learning rate: {self.args.lr}\n")
            f.write(f"  Early stopping patience: {self.args.patience}\n")
            f.write(f"  Weight decay: {self.args.weight_decay}\n")
            f.write(f"  Dataset directory: {self.args.data_dir}\n\n")

            f.write(f"Optimization configuration:\n")
            f.write(f"  Spleen weight: 40:1\n")
            f.write(f"  Random seed: 42\n")
            f.write(f"  Training epochs: 150\n")
            f.write(f"  Early stopping patience: 30\n\n")

            f.write(f"Dataset statistics:\n")
            f.write(f"  Training set size: {len(self.train_df)} samples\n")
            f.write(f"  Validation set size: {len(self.val_df)} samples\n")
            f.write(f"  Test set size: {len(self.test_df)} samples\n\n")

            f.write(f"Training results:\n")
            f.write(f"  Best validation AUC: {self.best_val_auc:.4f} (epoch {self.best_epoch})\n")
            if self.history['train_loss']:
                f.write(f"  Final training loss: {self.history['train_loss'][-1]:.4f}\n")
            if self.history['val_loss']:
                f.write(f"  Final validation loss: {self.history['val_loss'][-1]:.4f}\n\n")

            f.write(f"Test set results:\n")
            f.write(f"  Test set mean AUC: {test_auc:.4f}\n")
            f.write(f"  Heart: AUC = {test_auc_scores[0]:.4f}\n")
            f.write(f"  Lung: AUC = {test_auc_scores[1]:.4f}\n")
            f.write(f"  Spleen: AUC = {test_auc_scores[2]:.4f}\n")
            f.write(f"  Liver: AUC = {test_auc_scores[3]:.4f}\n")
            f.write(f"  Kidney: AUC = {test_auc_scores[4]:.4f}\n\n")

            f.write(f"Output files:\n")
            f.write(f"  Model file: {os.path.join(self.optimized_output_dir, 'resnet101_5class_best_optimized.pth')}\n")
            f.write(f"  Plot files: {self.optimized_output_dir}/*.png\n")
            f.write(f"  Log files: {self.logs_dir}/training_*.log\n")
            f.write(f"  Summary file: {summary_path}\n")

        self.logger.info(f"Training summary saved to: {summary_path}")

    def train(self):
        if self.args.mode == 'fast':
            epochs = 10
            patience = 5
        elif self.args.mode == 'medium':
            epochs = 30
            patience = 15
        else:
            epochs = 150
            patience = 30

        self.logger.info(f"\nStarting training, mode: {self.args.mode}")
        self.logger.info(f"Max epochs: {epochs}, Early stopping patience: {patience}")
        self.logger.info(f"Learning rate scheduler: ReduceLROnPlateau (patience 10 epochs)")
        self.logger.info(f"Stability optimizations: random seed set, more thorough training")

        best_auc = 0.0
        patience_counter = 0
        previous_lr = self.optimizer.param_groups[0]['lr']

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"\n[Epoch {epoch}] Learning rate: {current_lr:.6f}")

            train_loss = self.train_epoch(epoch, use_mixup=False)
            val_loss, val_auc, auc_scores, _, _, _ = self.validate(self.val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            self.history['val_auc_scores'].append(auc_scores)
            self.history['lr'].append(current_lr)

            epoch_time = time.time() - epoch_start_time

            self.logger.info(f"[Epoch {epoch}] Train loss: {train_loss:.4f}, "
                             f"Val loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            self.logger.info(f"Epoch time: {epoch_time:.1f} sec")
            self.logger.info(f"Per-class AUC: Heart={auc_scores[0]:.3f} Lung={auc_scores[1]:.3f} "
                             f"Spleen={auc_scores[2]:.3f} Liver={auc_scores[3]:.3f} Kidney={auc_scores[4]:.3f}")

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != previous_lr:
                    self.logger.info(f"Learning rate adjusted from {previous_lr:.6f} to {new_lr:.6f}")
                    previous_lr = new_lr
            else:
                self.scheduler.step()

            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                self.best_model_state = self.model.state_dict()
                self.best_val_auc = best_auc
                self.best_epoch = epoch

                checkpoint_path = os.path.join(self.optimized_output_dir, 'resnet101_5class_best_optimized.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_auc': best_auc,
                    'best_val_auc': self.best_val_auc,
                    'history': self.history,
                    'args': vars(self.args)
                }, checkpoint_path)
                self.logger.info(f"   Saved best model (AUC: {best_auc:.4f}) to {checkpoint_path}")
            else:
                patience_counter += 1
                self.logger.info(f"   No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    self.logger.info(f"\nEarly stopping triggered, stopping training")
                    break

        self.logger.info(f"\nTraining completed, best validation AUC: {best_auc:.4f}")
        self.logger.info(f"Training epochs: {epoch}/{epochs}")
        return best_auc

    def evaluate_test(self):
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("Starting test set evaluation")
        self.logger.info(f"{'=' * 60}")

        checkpoint_path = os.path.join(self.optimized_output_dir, 'resnet101_5class_best_optimized.pth')
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Loaded best model for testing (validation AUC: {checkpoint['best_val_auc']:.4f})")
            except Exception as e:
                self.logger.warning(f"Failed to load model: {e}")
                if self.best_model_state is not None:
                    self.model.load_state_dict(self.best_model_state)
                    self.logger.info(f"Using in-memory best model for evaluation (validation AUC: {self.best_val_auc:.4f})")
                else:
                    self.logger.info("No saved model state, using current model for evaluation")
        elif self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info(f"Using in-memory best model for evaluation (validation AUC: {self.best_val_auc:.4f})")
        else:
            self.logger.info("No saved model state, using current model for evaluation")

        test_loss, test_auc, test_auc_scores, test_preds, test_labels, test_outputs = self.validate(self.test_loader)

        self.logger.info(f"\n=== Test Set Evaluation Results ===")
        self.logger.info(f"Test loss: {test_loss:.4f}")
        self.logger.info(f"Test mean AUC: {test_auc:.4f}")
        self.logger.info(f"Heart: AUC = {test_auc_scores[0]:.4f}")
        self.logger.info(f"Lung: AUC = {test_auc_scores[1]:.4f}")
        self.logger.info(f"Spleen: AUC = {test_auc_scores[2]:.4f}")
        self.logger.info(f"Liver: AUC = {test_auc_scores[3]:.4f}")
        self.logger.info(f"Kidney: AUC = {test_auc_scores[4]:.4f}")

        self.history['test_auc'].append(test_auc)
        self.history['test_auc_scores'].append(test_auc_scores)

        self.generate_all_plots(test_preds, test_labels)

        return test_auc, test_auc_scores


def parse_args():
    parser = argparse.ArgumentParser(description='Optimized Tongue Diagnosis Model (with full output features)')

    parser.add_argument('--data_dir', type=str,
                        default=r'D:\tongue_dataset\HongKongPolyU\TongueDx2\tongue-fixed\datasets',
                        help='Dataset directory path')

    parser.add_argument('--mode', type=str, default='fast',
                        choices=['fast', 'medium', 'full'],
                        help='Debug mode: fast (quick), medium (medium), full (complete)')

    parser.add_argument('--optim', type=str, default='baseline',
                        choices=['baseline', 'attention_cbam', 'attention_se',
                                 'focal_loss', 'dice_loss', 'mixup'],
                        help='Optimization scheme selection')

    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default auto-set based on mode)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default auto-set based on mode)')
    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping patience (default auto-set based on mode)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage (even if GPU is available)')

    return parser.parse_args()


def main():
    set_seed(42, verbose=True)

    args = parse_args()

    if args.mode == 'fast':
        if args.batch_size is None:
            args.batch_size = 32
        if args.lr is None:
            args.lr = 0.001
        if args.patience is None:
            args.patience = 5
    elif args.mode == 'medium':
        if args.batch_size is None:
            args.batch_size = 16
        if args.lr is None:
            args.lr = 0.0005
        if args.patience is None:
            args.patience = 15
    else:
        if args.batch_size is None:
            args.batch_size = 12
        if args.lr is None:
            args.lr = 0.0001
        if args.patience is None:
            args.patience = 30

    print(f"\n{'=' * 60}")
    print(f"Optimized Tongue Diagnosis Model (with full output features)")
    print(f"Mode: {args.mode}")
    print(f"Optimization scheme: {args.optim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Experiment setup: Spleen negative weight 40:1 (best configuration)")
    print(f"Learning rate scheduler: ReduceLROnPlateau (patience 10 epochs)")
    print(f"Training stability: random seed set for reproducibility")
    print(f"Training completeness: full mode epochs 150, early stopping patience 30")
    print(f"Output directory: optimized_output_YYYYMMDD_HHMMSS/")
    print(f"Output files: model file + 6 plots + logs + summary")
    print(f"{'=' * 60}\n")

    trainer = TongueModelTrainer(args)
    best_val_auc = trainer.train()
    test_auc, test_auc_scores = trainer.evaluate_test()
    trainer.save_training_summary(test_auc, test_auc_scores)

    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Mode: {args.mode}")
    print(f"Optimization: {args.optim}")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Test mean AUC: {test_auc:.4f}")
    print(f"Output directory: {trainer.optimized_output_dir}")
    print(f"Stability optimizations: random seed 42, early stopping patience 30, training epochs 150")
    print(f"{'=' * 60}")


