import os
import sys
import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from itertools import cycle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

CLASS_NAMES = ['Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']
NUM_CLASSES = len(CLASS_NAMES)


class TongueDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        print(f"CSV columns: {list(self.data.columns)}")
        print(f"First row dtypes: {self.data.dtypes}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = None
        if 'image_path' in self.data.columns:
            img_path = self.data.iloc[idx]['image_path']
        elif 'image' in self.data.columns:
            img_path = self.data.iloc[idx]['image']
        elif 'img' in self.data.columns:
            img_path = self.data.iloc[idx]['img']
        elif 'path' in self.data.columns:
            img_path = self.data.iloc[idx]['path']
        else:
            img_path = self.data.iloc[idx, 0]

        if not isinstance(img_path, str):
            img_path = str(img_path)

        img_name = os.path.join(self.img_dir, img_path)
        image = Image.open(img_name).convert('RGB')

        labels = None
        if all(col in self.data.columns for col in CLASS_NAMES):
            labels = self.data.iloc[idx][CLASS_NAMES].values.astype(np.float32)
        else:
            labels = self.data.iloc[idx, 1:6].values.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class SEAttention(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class TongueModel5Class(torch.nn.Module):
    def __init__(self, resnet_version='resnet101', pretrained=False, use_attention=True, attention_type='se'):
        super(TongueModel5Class, self).__init__()

        self.base_model = torchvision.models.resnet101(weights=None)
        num_ftrs = self.base_model.fc.in_features

        self.use_attention = use_attention
        self.attention_type = attention_type

        if use_attention:
            if attention_type == 'se':
                self.attention1 = SEAttention(256)
                self.attention2 = SEAttention(512)
                self.attention3 = SEAttention(1024)
                self.attention4 = SEAttention(2048)

        self.base_model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(num_ftrs, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 5)
        )

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


class ModelLoader:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint = None
        self.model = None
        self.history = None

    def load_checkpoint(self):
        try:
            self.checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
            print(f"Loaded checkpoint: {self.checkpoint_path}")
            print(f"Best AUC: {self.checkpoint.get('best_auc', 'N/A')}")
            print(f"Best Val AUC: {self.checkpoint.get('best_val_auc', 'N/A')}")
            self.history = self.checkpoint.get('history', {})

            print("\nHistory keys:")
            for key in self.history.keys():
                print(f"  - {key}")
                if isinstance(self.history[key], list):
                    print(f"    Length: {len(self.history[key])}")
                    if key == 'val_auc_scores' and len(self.history[key]) > 0:
                        print(f"    First element type: {type(self.history[key][0])}")
                        if isinstance(self.history[key][0], (list, tuple)):
                            print(f"    First element length: {len(self.history[key][0])}")
                            print(f"    First element values: {self.history[key][0]}")

            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def build_model(self):
        if not self.checkpoint:
            raise ValueError("Checkpoint not loaded")

        args = self.checkpoint.get('args', {})
        use_attention = args.get('attention', True)
        attention_type = args.get('attention_type', 'se')

        self.model = TongueModel5Class(
            resnet_version='resnet101',
            pretrained=False,
            use_attention=use_attention,
            attention_type=attention_type
        )

        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        print("Model built successfully")
        return self.model


class DataLoaderManager:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, 'seg')
        self.list_dir = os.path.join(data_dir, 'list_new')

    def load_datasets(self):
        train_csv = os.path.join(self.list_dir, 'train.csv')
        val_csv = os.path.join(self.list_dir, 'val.csv')
        test_csv = os.path.join(self.list_dir, 'test.csv')

        datasets = {
            'train': TongueDataset(train_csv, self.img_dir, test_transform),
            'val': TongueDataset(val_csv, self.img_dir, test_transform),
            'test': TongueDataset(test_csv, self.img_dir, test_transform)
        }

        loaders = {
            'train': DataLoader(datasets['train'], batch_size=32, shuffle=False),
            'val': DataLoader(datasets['val'], batch_size=32, shuffle=False),
            'test': DataLoader(datasets['test'], batch_size=32, shuffle=False)
        }

        return datasets, loaders

    def load_csv_data(self):
        train_csv = os.path.join(self.list_dir, 'train.csv')
        val_csv = os.path.join(self.list_dir, 'val.csv')
        test_csv = os.path.join(self.list_dir, 'test.csv')

        data = {
            'train': pd.read_csv(train_csv),
            'val': pd.read_csv(val_csv),
            'test': pd.read_csv(test_csv)
        }

        return data


class InferenceEngine:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def run_inference(self):
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds)
                all_probs.append(probs)

        all_labels = np.vstack(all_labels)
        all_preds = np.vstack(all_preds)
        all_probs = np.vstack(all_probs)

        return all_labels, all_preds, all_probs


class MetricsCalculator:
    def __init__(self, labels, preds, probs):
        self.labels = labels
        self.preds = preds
        self.probs = probs

    def compute_confusion_matrices(self):
        matrices = []
        for i in range(NUM_CLASSES):
            cm = confusion_matrix(self.labels[:, i], self.preds[:, i])
            matrices.append(cm)
        return matrices

    def compute_roc_auc(self):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(NUM_CLASSES):
            fpr[i], tpr[i], _ = roc_curve(self.labels[:, i], self.probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(self.labels.ravel(), self.probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        return fpr, tpr, roc_auc


class PlotGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_auc_curves(self, history, filename='auc_curves.png'):
        plt.figure(figsize=(10, 6))

        colors = ['blue', 'green', 'red', 'purple', 'yellow', 'black']
        linestyles = ['-', '-', '-', '-', '-', '--']
        markers = ['o', '^', 's', 'D', 'v', '']

        if 'val_auc_scores' in history:
            auc_scores = history['val_auc_scores']
            if isinstance(auc_scores, list) and len(auc_scores) > 0:
                class_auc_data = {}
                for i, class_name in enumerate(CLASS_NAMES):
                    class_auc_data[class_name] = []

                for epoch_scores in auc_scores:
                    if isinstance(epoch_scores, (list, tuple)) and len(epoch_scores) >= len(CLASS_NAMES):
                        for i, class_name in enumerate(CLASS_NAMES):
                            score = float(epoch_scores[i])
                            class_auc_data[class_name].append(score)

                for i, class_name in enumerate(CLASS_NAMES):
                    if len(class_auc_data[class_name]) > 0:
                        plt.plot(class_auc_data[class_name], color=colors[i],
                                 linestyle=linestyles[i], linewidth=1.5,
                                 marker=markers[i], markersize=3,
                                 label=f'{class_name} AUC')

        if 'val_auc' in history:
            plt.plot(history['val_auc'], color=colors[-1], linestyle=linestyles[-1],
                     linewidth=2, label='Average AUC')

        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('AUC Curves During Training')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='lower right', frameon=True, shadow=False)
        plt.ylim(0, 1.0)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"AUC curves saved to: {output_path}")

    def plot_class_distribution(self, data_dict, filename='class_distribution.png'):
        plt.figure(figsize=(12, 6))

        distributions = {}
        for split, df in data_dict.items():
            if all(col in df.columns for col in CLASS_NAMES):
                distributions[split] = df[CLASS_NAMES].sum().values.astype(int)
            else:
                distributions[split] = df.iloc[:, 1:6].sum().values.astype(int)

        x = np.arange(len(CLASS_NAMES))
        width = 0.25

        colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
        labels = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}

        for i, (split, counts) in enumerate(distributions.items()):
            counts = counts.astype(int)
            plt.bar(x + i * width, counts, width=width, label=labels[split], color=colors[split])

        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution Across Datasets (Positive Samples)')
        plt.xticks(x + width, CLASS_NAMES)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.legend(loc='upper right', frameon=True, shadow=False)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"Class distribution saved to: {output_path}")

    def plot_confusion_matrices(self, matrices, filename='confusion_matrices.png'):
        fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(15, 3))

        for i, (class_name, cm) in enumerate(zip(CLASS_NAMES, matrices)):
            ax = axes[i]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        cbar=False, annot_kws={'fontsize': 9})
            ax.set_title(class_name)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['Negative', 'Positive'])
            ax.set_yticklabels(['Negative', 'Positive'])

        plt.suptitle('Confusion Matrices (Threshold=0.5)', fontsize=14, y=1.05)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrices saved to: {output_path}")

    def plot_loss_curves(self, history, filename='loss_curves.png'):
        plt.figure(figsize=(10, 6))

        if 'train_loss' in history:
            plt.plot(history['train_loss'], color='blue', linewidth=1.5, label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], color='orange', linewidth=1.5, label='Validation Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right', frameon=True, shadow=False)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"Loss curves saved to: {output_path}")

    def plot_prediction_distributions(self, labels, probs, filename='prediction_distributions.png'):
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes = axes.flatten()

        for i, class_name in enumerate(CLASS_NAMES):
            ax = axes[i]

            positive_probs = probs[labels[:, i] == 1, i]
            negative_probs = probs[labels[:, i] == 0, i]

            ax.hist(negative_probs, bins=20, range=(0, 1), color='red', alpha=0.7, label='Negative')
            ax.hist(positive_probs, bins=20, range=(0, 1), color='blue', alpha=0.7, label='Positive')

            ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.7)

            ax.set_xlabel('Prediction Probability')
            ax.set_ylabel('Count')
            ax.set_title(f'{class_name}')
            ax.set_xlim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper right', frameon=True, shadow=False)

        plt.suptitle('Prediction Distributions for All Classes', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"Prediction distributions saved to: {output_path}")

    def plot_roc_curves(self, fpr, tpr, roc_auc, filename='roc_curves.png'):
        plt.figure(figsize=(10, 8))

        colors = cycle(['blue', 'green', 'red', 'purple', 'yellow'])

        for i, color in zip(range(NUM_CLASSES), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=1,
                     label=f'{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})')

        plt.plot(fpr['micro'], tpr['micro'], color='black', linestyle='--', lw=2,
                 label=f'Average (Micro) (AUC = {roc_auc["micro"]:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Classes')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='lower right', frameon=True, shadow=False)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"ROC curves saved to: {output_path}")


class TonguePlotGenerator:
    def __init__(self, checkpoint_path, data_dir, output_dir):
        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        self.output_dir = output_dir

        self.model_loader = ModelLoader(checkpoint_path)
        self.data_manager = DataLoaderManager(data_dir)
        self.plot_generator = PlotGenerator(output_dir)

    def run(self):
        if not self.model_loader.load_checkpoint():
            return False

        model = self.model_loader.build_model()

        datasets, loaders = self.data_manager.load_datasets()
        csv_data = self.data_manager.load_csv_data()

        test_loader = loaders['test']
        inference_engine = InferenceEngine(model, test_loader)
        labels, preds, probs = inference_engine.run_inference()

        metrics = MetricsCalculator(labels, preds, probs)
        confusion_matrices = metrics.compute_confusion_matrices()
        fpr, tpr, roc_auc = metrics.compute_roc_auc()

        history = self.model_loader.history

        self.plot_generator.plot_auc_curves(history, 'auc_curves.png')
        self.plot_generator.plot_class_distribution(csv_data, 'class_distribution.png')
        self.plot_generator.plot_confusion_matrices(confusion_matrices, 'confusion_matrices.png')
        self.plot_generator.plot_loss_curves(history, 'loss_curves.png')
        self.plot_generator.plot_prediction_distributions(labels, probs, 'prediction_distributions.png')
        self.plot_generator.plot_roc_curves(fpr, tpr, roc_auc, 'roc_curves.png')

        print("\nAll plots generated successfully!")
        print(f"Output directory: {self.output_dir}")
        return True



    CHECKPOINT_PATH = "D:\\tongue_dataset\\HongKongPolyU\\TongueDx2\\tongue-fixed\\optimized_output_20260125_012014\\resnet101_5class_best_optimized.pth"
    DATA_DIR = "D:\\tongue_dataset\\HongKongPolyU\\TongueDx2\\tongue-fixed\\datasets"
    OUTPUT_DIR = "D:\\tongue_dataset\\HongKongPolyU\\TongueDx2\\tongue-fixed\\generated_plots"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generator = TonguePlotGenerator(CHECKPOINT_PATH, DATA_DIR, OUTPUT_DIR)
    generator.run()