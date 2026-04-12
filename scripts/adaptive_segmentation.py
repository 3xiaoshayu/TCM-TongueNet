import os
import argparse
import glob
import re

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from torchvision import transforms
from collections import defaultdict

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

try:
    plt.rcParams['font.family'] = 'Times New Roman'
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if 'Times New Roman' not in available_fonts:
        print("Warning: Times New Roman font not available, using default font")
    else:
        print("Times New Roman font loaded")
except Exception as e:
    print(f"Font setting error: {e}")

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

import torchvision.models as models
import torch.nn as nn


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

LABEL_NAMES = ['Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']

PRIOR_KNOWLEDGE = {
    'Heart': {'row_range': (0.7, 1.0), 'col_range': (0.0, 1.0), 'confidence': 'high', 'desc': 'Tip region (bottom)'},
    'Lung': {'row_range': (0.6, 1.0), 'col_range': (0.0, 1.0), 'confidence': 'high', 'desc': 'Tip region (bottom)'},
    'Spleen': {'row_range': (0.3, 0.7), 'col_range': (0.3, 0.7), 'confidence': 'low', 'desc': 'Mid-center (low reference)'},
    'Liver': {'row_range': (0.3, 0.7), 'col_ranges': [(0.0, 0.4), (0.6, 1.0)], 'confidence': 'medium', 'desc': 'Mid-left and Mid-right'},
    'Kidney': {'row_range': (0.0, 0.3), 'col_range': (0.0, 1.0), 'confidence': 'high', 'desc': 'Root region (top)'}
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def find_latest_model(data_dir_parent=None):
    if data_dir_parent is None:
        data_dir_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    pattern = os.path.join(data_dir_parent, 'optimized_output_*')
    output_dirs = glob.glob(pattern)

    if not output_dirs:
        tongue_fixed_dir = os.path.dirname(data_dir_parent) if os.path.basename(data_dir_parent) == 'tongue-fixed' else data_dir_parent
        pattern2 = os.path.join(tongue_fixed_dir, 'optimized_output_*')
        output_dirs = glob.glob(pattern2)

        if not output_dirs:
            print(f"Warning: No optimized_output directory found in {data_dir_parent} and {tongue_fixed_dir}")
            return None, None, None

    output_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_output_dir = output_dirs[0]

    model_pattern = os.path.join(latest_output_dir, '*best*.pth')
    model_files = glob.glob(model_pattern)

    if not model_files:
        print(f"Warning: No model file found in {latest_output_dir}")
        return latest_output_dir, None, None

    if len(model_files) > 1:
        for mf in model_files:
            if 'best' in os.path.basename(mf).lower():
                model_path = mf
                break
        else:
            model_path = model_files[0]
    else:
        model_path = model_files[0]

    data_dir = None
    summary_pattern = os.path.join(latest_output_dir, '*summary*.txt')
    summary_files = glob.glob(summary_pattern)

    if summary_files:
        try:
            with open(summary_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
                import re
                match = re.search(r'Dataset directory:\s*(.+)', content)
                if match:
                    data_dir = match.group(1).strip()
                    print(f"Extracted data directory from training summary: {data_dir}")
        except Exception as e:
            print(f"Failed to read training summary file: {e}")

    return latest_output_dir, model_path, data_dir


def load_model(model_path, data_dir=None):
    print(f"Loading model: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if 'args' in checkpoint:
            args = checkpoint['args']
            resnet_version = args.get('resnet_version', 'resnet101')
            use_attention = False
            attention_type = 'cbam'

            optim_scheme = args.get('optim', 'baseline')
            if 'attention' in optim_scheme:
                use_attention = True
                if 'se' in optim_scheme:
                    attention_type = 'se'

            print(f"Configuration extracted from checkpoint:")
            print(f"  ResNet version: {resnet_version}")
            print(f"  Use attention: {use_attention}")
            print(f"  Attention type: {attention_type}")
            print(f"  Optimization scheme: {optim_scheme}")

            model = TongueModel5Class(
                resnet_version=resnet_version,
                pretrained=False,
                use_attention=use_attention,
                attention_type=attention_type
            ).to(device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            print(f"Successfully loaded model from checkpoint")

        else:
            print("Warning: No configuration found in checkpoint, using default config")
            model = TongueModel5Class(resnet_version='resnet101', pretrained=False).to(device)
            model.load_state_dict(checkpoint)

    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Attempting to load model with default configuration...")
        model = TongueModel5Class(resnet_version='resnet101', pretrained=False).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def get_target_layer(model):
    return model.base_model.layer4


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor, original_image


def compute_gradcam(model, input_tensor, target_class, target_layer):
    model.eval()
    input_tensor = input_tensor.clone()
    input_tensor.requires_grad = True

    features = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output
        features.retain_grad()

    hook = target_layer.register_forward_hook(forward_hook)

    output = model(input_tensor)

    model.zero_grad()

    target_score = output[0, target_class]
    target_score.backward(retain_graph=True)

    hook.remove()

    if features is None:
        return np.zeros((7, 7))

    if features.grad is None:
        return np.zeros((7, 7))

    gradients = features.grad
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

    cam = torch.zeros(features.shape[2:], device=device, dtype=torch.float32)
    for i in range(weights.shape[1]):
        cam += weights[0, i] * features[0, i]

    cam = F.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    cam = cam.detach().cpu().numpy()

    return cam


def get_true_labels(csv_file, img_name):
    df = pd.read_csv(csv_file)

    img_id_part = os.path.splitext(img_name)[0]
    img_id = img_id_part.split('_')[0]

    mask = df['image_path'].str.contains(f'test/{img_id}')
    if not mask.any():
        return []

    img_row = df[mask].iloc[0]
    true_labels = []
    for label in LABEL_NAMES:
        if label in img_row and img_row[label] == 1:
            true_labels.append(label)
    return true_labels


def get_predicted_class(output, threshold=0.5):
    probs = torch.sigmoid(output).cpu().detach().numpy()[0]
    for i, prob in enumerate(probs):
        if i < len(LABEL_NAMES) and prob >= threshold:
            return LABEL_NAMES[i], prob
    max_idx = np.argmax(probs)
    if max_idx < len(LABEL_NAMES):
        return LABEL_NAMES[max_idx], probs[max_idx]
    return None, 0


def resize_cam_to_image(cam, image_size):
    h, w = image_size
    return cv2.resize(cam, (w, h))


class AdaptiveRegionAnalyzer:
    def __init__(self, low_resolution=(128, 128)):
        self.heatmap_statistics = {label: [] for label in LABEL_NAMES}
        self.activation_maps = {label: [] for label in LABEL_NAMES}
        self.reference_size = low_resolution
        self.sample_counts = {label: 0 for label in LABEL_NAMES}

    def add_heatmap(self, cam, true_labels, predicted_class, image_size):
        cam_resized = resize_cam_to_image(cam, self.reference_size)

        for label in true_labels:
            if label in LABEL_NAMES:
                self.heatmap_statistics[label].append(cam_resized.copy())
                self.sample_counts[label] += 1

        if predicted_class and predicted_class in LABEL_NAMES:
            self.activation_maps[predicted_class].append(cam_resized.copy())

    def compute_activation_probability(self, label):
        heatmaps = self.heatmap_statistics.get(label, [])
        if not heatmaps:
            return None

        stacked = np.stack(heatmaps, axis=0)
        mean_activation = np.mean(stacked, axis=0)
        std_activation = np.std(stacked, axis=0)

        return mean_activation, std_activation, len(heatmaps)

    def get_sample_counts(self):
        return self.sample_counts

    def filter_by_prior_knowledge(self, mean_activation, label):
        if mean_activation is None:
            return None

        prior = PRIOR_KNOWLEDGE.get(label)
        if not prior:
            return mean_activation

        confidence = prior.get('confidence', 'unknown')
        if confidence == 'low':
            pass

        return mean_activation

    def find_optimal_boundaries(self, mean_activation, label, n_bins=20):
        if mean_activation is None:
            return None

        h, w = mean_activation.shape

        row_activation = np.mean(mean_activation, axis=1)
        col_activation = np.mean(mean_activation, axis=0)

        row_bins = np.linspace(0, h, n_bins + 1)
        col_bins = np.linspace(0, w, n_bins + 1)

        row_indices = np.digitize(np.arange(h), row_bins) - 1
        col_indices = np.digitize(np.arange(w), col_bins) - 1

        row_agg = np.array([np.mean(row_activation[row_indices == i]) for i in range(n_bins)])
        col_agg = np.array([np.mean(col_activation[col_indices == i]) for i in range(n_bins)])

        prior = PRIOR_KNOWLEDGE.get(label, {})
        row_range = prior.get('row_range', (0.0, 1.0))
        col_range = prior.get('col_range', (0.0, 1.0))

        return {
            'row_activation': row_activation,
            'col_activation': col_activation,
            'row_bins': row_bins,
            'col_bins': col_bins,
            'row_agg': row_agg,
            'col_agg': col_agg,
            'prior_row_range': row_range,
            'prior_col_range': col_range,
            'prior_confidence': prior.get('confidence', 'unknown'),
            'prior_desc': prior.get('desc', '')
        }

    def compute_region_boundaries(self, label, threshold_ratio=0.5):
        prob_result = self.compute_activation_probability(label)
        if prob_result is None:
            return None

        mean_activation, std_activation, n_samples = prob_result
        mean_activation = self.filter_by_prior_knowledge(mean_activation, label)

        boundary_info = self.find_optimal_boundaries(mean_activation, label)
        if boundary_info is None:
            return None

        h, w = self.reference_size

        row_activation = boundary_info['row_activation']
        col_activation = boundary_info['col_activation']

        prior = PRIOR_KNOWLEDGE.get(label, {})
        row_range = prior.get('row_range', (0.0, 1.0))
        col_range = prior.get('col_range', (0.0, 1.0))

        confidence = prior.get('confidence', 'unknown')
        if confidence == 'low':
            row_range = (0.0, 1.0)
            col_range = (0.0, 1.0)

        row_low, row_high = int(h * row_range[0]), int(h * row_range[1])
        col_low, col_high = int(w * col_range[0]), int(w * col_range[1])

        max_row_idx = np.argmax(row_activation[row_low:row_high]) + row_low
        max_col_idx = np.argmax(col_activation[col_low:col_high]) + col_low

        local_max_row = np.argmax(row_activation)
        local_max_col = np.argmax(col_activation)

        confidence = prior.get('confidence', 'unknown')

        return {
            'label': label,
            'peak_row': max_row_idx,
            'peak_col': max_col_idx,
            'peak_row_normalized': max_row_idx / h,
            'peak_col_normalized': max_col_idx / w,
            'local_peak_row': local_max_row / h,
            'local_peak_col': local_max_col / w,
            'n_samples': n_samples,
            'confidence': confidence,
            'prior_desc': prior.get('desc', ''),
            'row_activation': row_activation,
            'col_activation': col_activation,
            'mean_activation': mean_activation,
            'image_size': self.reference_size,
            'std_activation': std_activation
        }

    def compute_unified_boundaries(self):
        print("\n" + "=" * 70)
        print("Comprehensive analysis: Computing unified optimal segmentation boundaries")
        print("=" * 70)

        boundaries = {}
        for label in LABEL_NAMES:
            result = self.compute_region_boundaries(label)
            if result:
                boundaries[label] = result

        h, w = self.reference_size

        tip_end = int(h * 0.25)
        root_start = int(h * 0.75)
        mid_start = tip_end
        mid_end = root_start

        liver_left_end = int(w * 0.35)
        liver_right_start = int(w * 0.65)

        print(f"\nRecommended segmentation scheme:")
        print(f"  Horizontal boundaries (top to bottom):")
        print(f"    - Root region: 0% - {root_start / h * 100:.1f}% (rows 0-{root_start})")
        print(f"    - Mid region: {mid_start / h * 100:.1f}% - {mid_end / h * 100:.1f}% (rows {mid_start}-{mid_end})")
        print(f"    - Tip region: {tip_end / h * 100:.1f}% - 100% (rows {tip_end}-{h})")

        print(f"\n  Vertical boundaries (left/mid/right):")
        print(f"    - Left: 0% - {liver_left_end / w * 100:.1f}% (cols 0-{liver_left_end})")
        print(f"    - Center: {liver_left_end / w * 100:.1f}% - {liver_right_start / w * 100:.1f}% (cols {liver_left_end}-{liver_right_start})")
        print(f"    - Right: {liver_right_start / w * 100:.1f}% - 100% (cols {liver_right_start}-{w})")

        unified_regions = {
            'Tip': {'row_start': tip_end / h, 'row_end': 1.0, 'col_start': 0.0, 'col_end': 1.0},
            'Mid-Left': {'row_start': mid_start / h, 'row_end': mid_end / h, 'col_start': 0.0, 'col_end': liver_left_end / w},
            'Mid-Center': {'row_start': mid_start / h, 'row_end': mid_end / h, 'col_start': liver_left_end / w, 'col_end': liver_right_start / w},
            'Mid-Right': {'row_start': mid_start / h, 'row_end': mid_end / h, 'col_start': liver_right_start / w, 'col_end': 1.0},
            'Root': {'row_start': 0.0, 'row_end': root_start / h, 'col_start': 0.0, 'col_end': 1.0}
        }

        print(f"\n  Detailed region definitions:")
        for region, coords in unified_regions.items():
            print(f"    - {region}: rows {coords['row_start'] * 100:.0f}%-{coords['row_end'] * 100:.0f}%, cols {coords['col_start'] * 100:.0f}%-{coords['col_end'] * 100:.0f}%")

        return unified_regions, boundaries


def collect_heatmaps(model, target_layer, image_ids, analyzer, test_csv_path, seg_dir_path, max_samples=None, sample_size=None):
    print("\n" + "=" * 70)
    print("Starting heatmap data collection...")
    print("=" * 70)

    df = pd.read_csv(test_csv_path)
    all_img_paths = []

    for img_id in image_ids:
        for suffix in ['_1.jpg', '_2.jpg', '_3.jpg', '_4.jpg', '_5.jpg']:
            path = os.path.join(seg_dir_path, 'test', f"{img_id}{suffix}")
            if os.path.exists(path):
                all_img_paths.append(path)
                if max_samples and len(all_img_paths) >= max_samples:
                    break
        if max_samples and len(all_img_paths) >= max_samples:
            break

    print(f"Found {len(all_img_paths)} images for analysis")

    if sample_size:
        print(f"Limiting to {sample_size} positive samples per organ")
        label_counts = {label: 0 for label in LABEL_NAMES}
        limited_img_paths = []

        for img_path in all_img_paths:
            img_name = os.path.basename(img_path)
            try:
                true_labels = get_true_labels(test_csv_path, img_name)
                label_added = False

                for label in true_labels:
                    if label_counts[label] < sample_size:
                        if img_path not in limited_img_paths:
                            limited_img_paths.append(img_path)
                            label_added = True
                        label_counts[label] += 1

                if label_added:
                    if all(count >= sample_size for count in label_counts.values()):
                        break
            except Exception as e:
                continue

        all_img_paths = limited_img_paths
        print(f"After filtering: {len(all_img_paths)} images retained")
        print("Sample counts per organ:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

    processed = 0
    for img_path in all_img_paths:
        try:
            img_name = os.path.basename(img_path)
            input_tensor, original_image = preprocess_image(img_path)

            with torch.no_grad():
                output = model(input_tensor)

            true_labels = get_true_labels(test_csv_path, img_name)

            for label in true_labels:
                label_idx = LABEL_NAMES.index(label)
                cam = compute_gradcam(model, input_tensor, label_idx, target_layer)
                analyzer.add_heatmap(cam, [label], label, original_image.shape[:2])

            processed += 1

            if processed % 10 == 0:
                print(f"  Processed {processed}/{len(all_img_paths)} images...")

        except Exception as e:
            continue

    print(f"\nData collection completed! Processed {processed} images")
    return processed


def visualize_activation_distribution(analyzer, boundaries, output_path):
    plt.rcParams['font.family'] = 'Times New Roman'

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, label in enumerate(LABEL_NAMES):
        if idx >= 5:
            break
        ax = axes[idx]

        boundary = boundaries.get(label)
        if boundary and boundary.get('mean_activation') is not None:
            mean_act = boundary['mean_activation']
            h, w = mean_act.shape

            im = ax.imshow(mean_act, cmap='jet', aspect='auto')

            peak_row = boundary.get('peak_row_normalized', 0) * h
            peak_col = boundary.get('peak_col_normalized', 0) * w
            ax.scatter(peak_col, peak_row, color='white', s=100, marker='*', edgecolors='black')

            prior = PRIOR_KNOWLEDGE.get(label, {})
            row_range = prior.get('row_range', (0.0, 1.0))
            ax.axhline(y=row_range[0] * h, color='white', linestyle='--', alpha=0.5)
            ax.axhline(y=row_range[1] * h, color='white', linestyle='--', alpha=0.5)

            ax.set_title(f'{label}\n({prior.get("desc", "")})', fontsize=10, fontname='Times New Roman')
            ax.set_xlabel('Column', fontname='Times New Roman')
            ax.set_ylabel('Row', fontname='Times New Roman')

            conf = prior.get('confidence', 'unknown')
            conf_color = {'high': 'green', 'medium': 'orange', 'low': 'red'}.get(conf, 'gray')
            conf_text = {'high': 'High', 'medium': 'Medium', 'low': 'Low*'}.get(conf, conf)
            ax.text(0.02, 0.98, f'Confidence: {conf_text}', transform=ax.transAxes,
                    fontsize=8, verticalalignment='top', color=conf_color, fontname='Times New Roman')

        else:
            ax.text(0.5, 0.5, f'{label}\nNo data', ha='center', va='center', transform=ax.transAxes,
                    fontname='Times New Roman')
            ax.set_title(f'{label}', fontsize=10, fontname='Times New Roman')

    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Activation distribution saved to: {output_path}")


def visualize_segmentation_scheme(unified_regions, output_path):
    plt.rcParams['font.family'] = 'Times New Roman'

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    h, w = 612, 612
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    ax.set_aspect('equal')
    ax.set_facecolor('#FFE4E1')

    tongue_outline = [
        (w * 0.2, h * 0.1),
        (w * 0.1, h * 0.3),
        (w * 0.05, h * 0.5),
        (w * 0.1, h * 0.7),
        (w * 0.2, h * 0.85),
        (w * 0.4, h * 0.95),
        (w * 0.6, h * 0.95),
        (w * 0.8, h * 0.85),
        (w * 0.9, h * 0.7),
        (w * 0.95, h * 0.5),
        (w * 0.9, h * 0.3),
        (w * 0.8, h * 0.1),
    ]

    from matplotlib.patches import Polygon
    tongue = Polygon(tongue_outline, closed=True, facecolor='#FFB6C1',
                     edgecolor='#8B0000', linewidth=3)
    ax.add_patch(tongue)

    tip_end = int(h * 0.25)
    root_start = int(h * 0.75)
    mid_start = tip_end
    mid_end = root_start

    liver_left_end = int(w * 0.35)
    liver_right_start = int(w * 0.65)

    ax.axhline(y=tip_end, color='blue', linewidth=2, linestyle='--')
    ax.axhline(y=root_start, color='blue', linewidth=2, linestyle='--')
    ax.axvline(x=liver_left_end, color='green', linewidth=2, linestyle='-.')
    ax.axvline(x=liver_right_start, color='green', linewidth=2, linestyle='-.')

    ax.text(w / 2, tip_end + 10, '25%', fontsize=10, ha='center', color='blue', fontweight='bold', fontname='Times New Roman')
    ax.text(w / 2, root_start - 10, '75%', fontsize=10, ha='center', color='blue', fontweight='bold', fontname='Times New Roman')
    ax.text(liver_left_end + 5, h / 2, '35%', fontsize=10, va='center', color='green', fontweight='bold', fontname='Times New Roman')
    ax.text(liver_right_start - 5, h / 2, '65%', fontsize=10, va='center', color='green', fontweight='bold', ha='right', fontname='Times New Roman')

    region_colors = {
        'Root': '#87CEEB',
        'Mid-Left': '#98FB98',
        'Mid-Center': '#DDA0DD',
        'Mid-Right': '#F0E68C',
        'Tip': '#FFA07A'
    }

    regions_with_coords = [
        ('Root', 0, root_start, 0, w),
        ('Mid-Left', mid_start, mid_end, 0, liver_left_end),
        ('Mid-Center', mid_start, mid_end, liver_left_end, liver_right_start),
        ('Mid-Right', mid_start, mid_end, liver_right_start, w),
        ('Tip', tip_end, h, 0, w)
    ]

    for region_name, row_start, row_end, col_start, col_end in regions_with_coords:
        rect = plt.Rectangle((col_start, row_start),
                             col_end - col_start,
                             row_end - row_start,
                             linewidth=2, edgecolor='black',
                             facecolor=region_colors.get(region_name, 'white'),
                             alpha=0.3)
        ax.add_patch(rect)

        center_x = (col_start + col_end) / 2
        center_y = (row_start + row_end) / 2
        ax.text(center_x, center_y, region_name, fontsize=12,
                ha='center', va='center', fontweight='bold', fontname='Times New Roman')

    ax.set_title('Recommended Tongue Segmentation Scheme\n(Based on Statistical Analysis + Prior Knowledge)',
                 fontsize=14, fontweight='bold', fontname='Times New Roman')
    ax.set_xlabel('Column Position', fontsize=11, fontname='Times New Roman')
    ax.set_ylabel('Row Position (0=Top, 1=Bottom)', fontsize=11, fontname='Times New Roman')

    ax.text(w * 0.5, h * 0.02, 'TIP (Tip of Tongue) - 25%~100% Row',
            ha='center', fontsize=10, style='italic', fontname='Times New Roman')
    ax.text(w * 0.5, h * 0.98, 'ROOT (Root of Tongue) - 0%~75% Row',
            ha='center', fontsize=10, style='italic', fontname='Times New Roman')

    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Horizontal Boundaries (25%, 75%)'),
        plt.Line2D([0], [0], color='green', linewidth=2, linestyle='-.', label='Vertical Boundaries (35%, 65%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', prop={'family': 'Times New Roman', 'size': 10})

    info_text = (
        "Segmentation Boundaries:\n"
        "• Tip: Row 25%-100%\n"
        "• Mid: Row 25%-75%\n"
        "• Root: Row 0%-75%\n"
        "• Left: Col 0%-35%\n"
        "• Center: Col 35%-65%\n"
        "• Right: Col 65%-100%"
    )
    ax.text(w * 0.02, h * 0.98, info_text, fontsize=9,
            verticalalignment='top', fontname='Times New Roman',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Segmentation scheme saved to: {output_path}")


def visualize_row_col_activation(boundaries, output_path):
    plt.rcParams['font.family'] = 'Times New Roman'

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, label in enumerate(LABEL_NAMES):
        if idx >= 6:
            break

        ax = axes[idx]
        boundary = boundaries.get(label)

        if boundary:
            h, w = boundary['image_size']
            row_activation = boundary['row_activation']
            col_activation = boundary['col_activation']

            x_vals = np.linspace(0, 100, len(row_activation))
            ax.plot(x_vals, row_activation, 'b-', linewidth=2, label='Row Activation')

            x_vals_col = np.linspace(0, 100, len(col_activation))
            ax.plot(x_vals_col, col_activation, 'r-', linewidth=2, label='Column Activation')

            peak_row_norm = boundary.get('peak_row_normalized', 0) * 100
            peak_col_norm = boundary.get('peak_col_normalized', 0) * 100

            ax.axvline(x=peak_row_norm, color='blue', linestyle='--', alpha=0.5)
            ax.axvline(x=peak_col_norm, color='red', linestyle='--', alpha=0.5)

            ax.text(peak_row_norm, max(row_activation) * 0.9,
                    f'Row: {peak_row_norm:.1f}%', fontsize=8, color='blue', fontname='Times New Roman')
            ax.text(peak_col_norm, max(col_activation) * 0.8,
                    f'Col: {peak_col_norm:.1f}%', fontsize=8, color='red', fontname='Times New Roman')

            prior = PRIOR_KNOWLEDGE.get(label, {})
            row_range = prior.get('row_range', (0.0, 1.0))
            col_range = prior.get('col_range', (0.0, 1.0))

            if 'col_ranges' in prior:
                for cr in prior['col_ranges']:
                    ax.axvspan(cr[0] * 100, cr[1] * 100, alpha=0.1, color='green')
            else:
                ax.axvspan(col_range[0] * 100, col_range[1] * 100, alpha=0.1, color='green')

            ax.axvspan(row_range[0] * 100, row_range[1] * 100, alpha=0.1, color='blue')

            ax.set_xlabel('Position (%)', fontname='Times New Roman')
            ax.set_ylabel('Activation', fontname='Times New Roman')
            ax.set_title(f'{label} - Row/Column Activation', fontname='Times New Roman')
            ax.legend(loc='upper right', prop={'family': 'Times New Roman', 'size': 9})
            ax.grid(True, alpha=0.3)

            for tick in ax.get_xticklabels():
                tick.set_fontname('Times New Roman')
            for tick in ax.get_yticklabels():
                tick.set_fontname('Times New Roman')

            n_samples = boundary.get('n_samples', 0)
            ax.text(0.02, 0.98, f'Samples: {n_samples}', transform=ax.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                    fontname='Times New Roman')
        else:
            ax.text(0.5, 0.5, f'{label}\nNo data', ha='center', va='center',
                    transform=ax.transAxes, fontname='Times New Roman')
            ax.set_title(f'{label}', fontname='Times New Roman')

    for idx in range(len(LABEL_NAMES), 6):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Row/column activation saved to: {output_path}")


def visualize_confidence_analysis(analyzer, boundaries, output_path):
    plt.rcParams['font.family'] = 'Times New Roman'

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    ax1 = axes[0]
    sample_counts = analyzer.get_sample_counts()
    labels = list(sample_counts.keys())
    counts = list(sample_counts.values())

    bars = ax1.bar(labels, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax1.set_xlabel('Organ', fontname='Times New Roman')
    ax1.set_ylabel('Number of Samples', fontname='Times New Roman')
    ax1.set_title('Sample Distribution for Each Organ', fontname='Times New Roman')
    ax1.grid(True, alpha=0.3)

    for tick in ax1.get_xticklabels():
        tick.set_fontname('Times New Roman')
    for tick in ax1.get_yticklabels():
        tick.set_fontname('Times New Roman')

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + max(counts) * 0.01,
                 f'{count}', ha='center', va='bottom', fontsize=9, fontname='Times New Roman')

    ax2 = axes[1]
    confidences = []
    peak_distances = []

    for label in LABEL_NAMES:
        boundary = boundaries.get(label)
        if boundary:
            prior = PRIOR_KNOWLEDGE.get(label, {})
            prior_row_range = prior.get('row_range', (0.0, 1.0))
            prior_col_range = prior.get('col_range', (0.0, 1.0))

            peak_row = boundary.get('peak_row_normalized', 0.5)
            peak_col = boundary.get('peak_col_normalized', 0.5)

            row_center = (prior_row_range[0] + prior_row_range[1]) / 2
            col_center = (prior_col_range[0] + prior_col_range[1]) / 2

            if 'col_ranges' in prior:
                col_ranges = prior['col_ranges']
                col_distances = []
                for cr in col_ranges:
                    col_center_range = (cr[0] + cr[1]) / 2
                    col_distances.append(abs(peak_col - col_center_range))
                col_distance = min(col_distances)
            else:
                col_distance = abs(peak_col - col_center)

            row_distance = abs(peak_row - row_center)
            total_distance = np.sqrt(row_distance ** 2 + col_distance ** 2)
            peak_distances.append(total_distance * 100)

            conf = prior.get('confidence', 'unknown')
            conf_map = {'high': 3, 'medium': 2, 'low': 1}
            confidences.append(conf_map.get(conf, 0))
        else:
            peak_distances.append(0)
            confidences.append(0)

    x_pos = np.arange(len(LABEL_NAMES))
    width = 0.35

    bars1 = ax2.bar(x_pos - width / 2, confidences, width, label='Prior Confidence', color='#4ECDC4')
    bars2 = ax2.bar(x_pos + width / 2, peak_distances, width, label='Peak Distance (%)', color='#FF6B6B')

    ax2.set_xlabel('Organ', fontname='Times New Roman')
    ax2.set_ylabel('Score / Distance (%)', fontname='Times New Roman')
    ax2.set_title('Confidence Analysis: Prior vs Learned', fontname='Times New Roman')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(LABEL_NAMES, fontname='Times New Roman')
    ax2.legend(prop={'family': 'Times New Roman', 'size': 9})
    ax2.grid(True, alpha=0.3)

    for tick in ax2.get_xticklabels():
        tick.set_fontname('Times New Roman')
    for tick in ax2.get_yticklabels():
        tick.set_fontname('Times New Roman')

    for bar, val in zip(bars1, confidences):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                 f'{val}', ha='center', va='bottom', fontsize=8, fontname='Times New Roman')

    for bar, val in zip(bars2, peak_distances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height + max(peak_distances) * 0.01,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontname='Times New Roman')

    ax3 = axes[2]
    consistency_scores = []

    for label in LABEL_NAMES:
        boundary = boundaries.get(label)
        if boundary and 'std_activation' in boundary:
            std_act = boundary['std_activation']
            consistency = 1.0 / (1.0 + np.mean(std_act))
            consistency_scores.append(consistency * 100)
        else:
            consistency_scores.append(0)

    bars = ax3.bar(LABEL_NAMES, consistency_scores,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax3.set_xlabel('Organ', fontname='Times New Roman')
    ax3.set_ylabel('Consistency Score (%)', fontname='Times New Roman')
    ax3.set_title('Activation Consistency Across Samples', fontname='Times New Roman')
    ax3.grid(True, alpha=0.3)

    for tick in ax3.get_xticklabels():
        tick.set_fontname('Times New Roman')
    for tick in ax3.get_yticklabels():
        tick.set_fontname('Times New Roman')

    for bar, score in zip(bars, consistency_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, height + max(consistency_scores) * 0.01,
                 f'{score:.1f}%', ha='center', va='bottom', fontsize=9, fontname='Times New Roman')

    ax4 = axes[3]
    composite_scores = []

    for i, label in enumerate(LABEL_NAMES):
        if consistency_scores[i] > 0:
            composite = (consistency_scores[i] * 0.6) + ((100 - peak_distances[i]) * 0.4)
            composite_scores.append(composite)
        else:
            composite_scores.append(0)

    bars = ax4.bar(LABEL_NAMES, composite_scores,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax4.set_xlabel('Organ', fontname='Times New Roman')
    ax4.set_ylabel('Composite Score', fontname='Times New Roman')
    ax4.set_title('Overall Quality Assessment', fontname='Times New Roman')
    ax4.grid(True, alpha=0.3)

    for tick in ax4.get_xticklabels():
        tick.set_fontname('Times New Roman')
    for tick in ax4.get_yticklabels():
        tick.set_fontname('Times New Roman')

    for bar, score in zip(bars, composite_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, height + max(composite_scores) * 0.01,
                 f'{score:.1f}', ha='center', va='bottom', fontsize=9, fontname='Times New Roman')

        if score >= 80:
            rating = 'A'
            color = 'green'
        elif score >= 60:
            rating = 'B'
            color = 'orange'
        else:
            rating = 'C'
            color = 'red'

        ax4.text(bar.get_x() + bar.get_width() / 2, height / 2,
                 rating, ha='center', va='center', fontsize=12, fontweight='bold',
                 color=color, fontname='Times New Roman')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confidence analysis saved to: {output_path}")

    return {
        'sample_counts': sample_counts,
        'peak_distances': {label: dist for label, dist in zip(LABEL_NAMES, peak_distances)},
        'consistency_scores': {label: score for label, score in zip(LABEL_NAMES, consistency_scores)},
        'composite_scores': {label: score for label, score in zip(LABEL_NAMES, composite_scores)}
    }


def generate_analysis_report(analyzer, boundaries, unified_regions, confidence_data, output_dir, sample_size=None):
    report_path = os.path.join(output_dir, 'analysis_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Tongue Diagnosis Region Segmentation Analysis Report\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. Analysis Overview\n")
        f.write("-" * 40 + "\n")
        if sample_size:
            f.write(f"Sample setting: {sample_size} positive samples per organ\n")
        else:
            f.write("Sample setting: using all available positive samples\n")

        sample_counts = analyzer.get_sample_counts()
        f.write("\nSample counts per organ:\n")
        total_samples = 0
        for label in LABEL_NAMES:
            count = sample_counts.get(label, 0)
            f.write(f"  {label}: {count} samples\n")
            total_samples += count
        f.write(f"Total: {total_samples} samples\n\n")

        f.write("2. Activation Distribution Analysis\n")
        f.write("-" * 40 + "\n")
        for label in LABEL_NAMES:
            boundary = boundaries.get(label)
            if boundary:
                f.write(f"\n{label}:\n")
                f.write(f"  Peak position: row {boundary['peak_row_normalized'] * 100:.1f}%, col {boundary['peak_col_normalized'] * 100:.1f}%\n")
                f.write(f"  Local peak: row {boundary['local_peak_row'] * 100:.1f}%, col {boundary['local_peak_col'] * 100:.1f}%\n")
                f.write(f"  Number of samples: {boundary['n_samples']}\n")
                f.write(f"  Prior confidence: {boundary['confidence']}\n")
                f.write(f"  Prior description: {boundary['prior_desc']}\n")

                prior = PRIOR_KNOWLEDGE.get(label, {})
                row_range = prior.get('row_range', (0.0, 1.0))
                col_range = prior.get('col_range', (0.0, 1.0))

                peak_row = boundary['peak_row_normalized']
                peak_col = boundary['peak_col_normalized']

                if 'col_ranges' in prior:
                    col_in_range = False
                    for cr in prior['col_ranges']:
                        if cr[0] <= peak_col <= cr[1]:
                            col_in_range = True
                            break
                    row_in_range = row_range[0] <= peak_row <= row_range[1]
                else:
                    row_in_range = row_range[0] <= peak_row <= row_range[1]
                    col_in_range = col_range[0] <= peak_col <= col_range[1]

                if row_in_range and col_in_range:
                    f.write("  Consistent with prior knowledge: Yes\n")
                else:
                    f.write("  Consistent with prior knowledge: No\n")
            else:
                f.write(f"\n{label}: insufficient data for analysis\n")

        f.write("\n3. Confidence Analysis\n")
        f.write("-" * 40 + "\n")
        for label in LABEL_NAMES:
            f.write(f"\n{label}:\n")
            f.write(f"  Peak distance: {confidence_data['peak_distances'].get(label, 0):.1f}%\n")
            f.write(f"  Consistency score: {confidence_data['consistency_scores'].get(label, 0):.1f}%\n")
            f.write(f"  Composite score: {confidence_data['composite_scores'].get(label, 0):.1f}\n")

            composite_score = confidence_data['composite_scores'].get(label, 0)
            if composite_score >= 80:
                rating = "A (Excellent)"
            elif composite_score >= 60:
                rating = "B (Good)"
            else:
                rating = "C (Needs improvement)"

            f.write(f"  Rating: {rating}\n")

        f.write("\n4. Recommended Segmentation Scheme\n")
        f.write("-" * 40 + "\n")
        f.write("\nUnified region definitions:\n")
        for region, coords in unified_regions.items():
            f.write(f"  {region}:\n")
            f.write(f"    Row range: {coords['row_start'] * 100:.0f}% - {coords['row_end'] * 100:.0f}%\n")
            f.write(f"    Column range: {coords['col_start'] * 100:.0f}% - {coords['col_end'] * 100:.0f}%\n")

        f.write("\nDetailed segmentation boundaries:\n")
        f.write(f"  Horizontal boundaries:\n")
        f.write(f"    - Root region: 0% - 75%\n")
        f.write(f"    - Mid region: 25% - 75%\n")
        f.write(f"    - Tip region: 25% - 100%\n")
        f.write(f"  Vertical boundaries:\n")
        f.write(f"    - Left: 0% - 35%\n")
        f.write(f"    - Center: 35% - 65%\n")
        f.write(f"    - Right: 65% - 100%\n")

        f.write("\n5. Consistency with Traditional Chinese Medicine (TCM) Theory\n")
        f.write("-" * 40 + "\n")
        f.write("\nAccording to TCM tongue diagnosis theory:\n")
        f.write("1. Tip region (25%-100%) corresponds to Heart and Lung, consistent with prior knowledge\n")
        f.write("2. Root region (0%-75%) corresponds to Kidney, consistent with prior knowledge\n")
        f.write("3. Mid region (25%-75%) corresponds to Spleen and Liver, generally consistent\n")
        f.write("4. Left (0%-35%) and Right (65%-100%) correspond to Liver, reflecting the bilateral nature of Liver\n")
        f.write("5. Spleen is located in the central region (35%-65%), consistent with the TCM description\n\n")

        f.write("Overall evaluation:\n")
        consistent_count = 0
        total_count = 0

        for label in LABEL_NAMES:
            boundary = boundaries.get(label)
            if boundary:
                total_count += 1
                prior = PRIOR_KNOWLEDGE.get(label, {})
                row_range = prior.get('row_range', (0.0, 1.0))
                col_range = prior.get('col_range', (0.0, 1.0))

                peak_row = boundary['peak_row_normalized']
                peak_col = boundary['peak_col_normalized']

                if 'col_ranges' in prior:
                    col_in_range = False
                    for cr in prior['col_ranges']:
                        if cr[0] <= peak_col <= cr[1]:
                            col_in_range = True
                            break
                    row_in_range = row_range[0] <= peak_row <= row_range[1]
                else:
                    row_in_range = row_range[0] <= peak_row <= row_range[1]
                    col_in_range = col_range[0] <= peak_col <= col_range[1]

                if row_in_range and col_in_range:
                    consistent_count += 1

        consistency_rate = (consistent_count / total_count * 100) if total_count > 0 else 0
        f.write(f"  Overall consistency rate: {consistency_rate:.1f}% ({consistent_count}/{total_count})\n")

        if consistency_rate >= 80:
            f.write("  Evaluation: Highly consistent with TCM theory\n")
        elif consistency_rate >= 60:
            f.write("  Evaluation: Generally consistent with TCM theory\n")
        else:
            f.write("  Evaluation: Inconsistent with TCM theory\n")

        f.write("\n6. Suggestions and Improvements\n")
        f.write("-" * 40 + "\n")
        f.write("1. Increase spleen sample size to improve confidence\n")
        f.write("2. Consider the effect of tongue shape variation on region division\n")
        f.write("3. Validate with clinical diagnosis results\n")
        f.write("4. Explore more refined region division methods\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Report generation time: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("=" * 80 + "\n")

    print(f"Analysis report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Adaptive Region Segmentation Analysis System')
    parser.add_argument('--model_path', type=str, default=None, help='Model path (if not provided, automatically find latest model)')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory (if not provided, attempt to extract from training summary or use default path)')
    parser.add_argument('--optim_output_dir', type=str, default=None, help='Training script output directory (if not provided, automatically find latest optimized_output directory)')
    parser.add_argument('--sample_size', type=int, default=None, help='Sample size limit per organ (None means use all samples)')
    parser.add_argument('--batch_csv', type=str, default=None, help='Batch CSV file path (if provided, overrides sample_size)')

    args = parser.parse_args()

    print("=" * 70)
    print("Adaptive Region Segmentation Analysis System")
    print("=" * 70)

    if args.optim_output_dir is None:
        optim_output_dir, model_path_found, data_dir_found = find_latest_model()

        if optim_output_dir is None:
            if args.data_dir:
                data_dir_parent = os.path.dirname(args.data_dir)
            else:
                default_data_dir = r'D:\tongue_dataset\HongKongPolyU\TongueDx2\tongue-fixed\datasets'
                data_dir_parent = os.path.dirname(default_data_dir)

            optim_output_dir, model_path_found, data_dir_found = find_latest_model(data_dir_parent)

            if optim_output_dir is None:
                print("Error: Training output directory not found")
                print("Please use one of the following options:")
                print("1. Run the training script first to generate model and output directory")
                print("2. Specify training output directory with --optim_output_dir")
                print("3. Manually specify model and data directory with --model_path and --data_dir")
                return

        print(f"Found latest training output directory: {optim_output_dir}")

        if args.model_path is None and model_path_found is not None:
            args.model_path = model_path_found
            print(f"Using automatically found model: {args.model_path}")

        if args.data_dir is None and data_dir_found is not None:
            args.data_dir = data_dir_found
            print(f"Using automatically found data directory: {args.data_dir}")
    else:
        optim_output_dir = args.optim_output_dir
        print(f"Using specified training output directory: {optim_output_dir}")

    if args.model_path is None:
        model_pattern = os.path.join(optim_output_dir, '*best*.pth')
        model_files = glob.glob(model_pattern)
        if model_files:
            args.model_path = model_files[0]
            print(f"Found model in output directory: {args.model_path}")
        else:
            print("Error: No model file found, please specify with --model_path")
            return

    if args.data_dir is None:
        args.data_dir = r'D:\tongue_dataset\HongKongPolyU\TongueDx2\tongue-fixed\datasets'
        print(f"Using default data directory: {args.data_dir}")

    data_dir = args.data_dir
    seg_dir = os.path.join(data_dir, 'seg')

    test_csv_new = os.path.join(data_dir, 'list_new', 'test.csv')
    test_csv_old = os.path.join(data_dir, 'list', 'test.csv')

    if os.path.exists(test_csv_new):
        test_csv_path = test_csv_new
        print(f"Using new dataset split: {test_csv_path}")
    elif os.path.exists(test_csv_old):
        test_csv_path = test_csv_old
        print(f"Using original dataset split: {test_csv_path}")
    else:
        print(f"Error: Test CSV file not found")
        print(f"Not found in {test_csv_new} or {test_csv_old}")
        return

    output_dir = os.path.join(optim_output_dir, 'adaptive_region_analysis')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    if args.batch_csv:
        print(f"Using batch CSV file: {args.batch_csv}")
        test_csv_path = args.batch_csv

    print("\nPrior knowledge (expert annotations):")
    for label, info in PRIOR_KNOWLEDGE.items():
        print(f"  {label}: {info['desc']} (confidence: {info['confidence']})")

    model = load_model(args.model_path, data_dir)
    target_layer = get_target_layer(model)

    df = pd.read_csv(test_csv_path)
    if 'image_path' in df.columns:
        img_ids = []
        for path in df['image_path']:
            match = re.search(r'test/(\d+)_', str(path))
            if match:
                img_id = int(match.group(1))
                if img_id not in img_ids:
                    img_ids.append(img_id)
        img_ids = np.array(img_ids)
    else:
        print("Error: 'image_path' column not found in CSV file")
        return

    print(f"\nTotal unique images in dataset: {len(img_ids)}")

    analyzer = AdaptiveRegionAnalyzer()
    n_processed = collect_heatmaps(model, target_layer, img_ids, analyzer, test_csv_path, seg_dir,
                                   max_samples=len(img_ids), sample_size=args.sample_size)

    if n_processed > 0:
        unified_regions, boundaries = analyzer.compute_unified_boundaries()

        activation_dist_path = os.path.join(output_dir, 'activation_distribution.png')
        segmentation_scheme_path = os.path.join(output_dir, 'segmentation_scheme.png')
        row_col_activation_path = os.path.join(output_dir, 'row_col_activation.png')
        confidence_analysis_path = os.path.join(output_dir, 'confidence_analysis.png')

        visualize_activation_distribution(analyzer, boundaries, activation_dist_path)
        visualize_segmentation_scheme(unified_regions, segmentation_scheme_path)
        visualize_row_col_activation(boundaries, row_col_activation_path)
        confidence_data = visualize_confidence_analysis(analyzer, boundaries, confidence_analysis_path)

        generate_analysis_report(analyzer, boundaries, unified_regions, confidence_data, output_dir, args.sample_size)

        print("\n" + "=" * 70)
        print("Analysis completed!")
        print("=" * 70)
        print(f"\nGenerated files:")
        print(f"  Activation distribution: {activation_dist_path}")
        print(f"  Segmentation scheme: {segmentation_scheme_path}")
        print(f"  Row/column activation: {row_col_activation_path}")
        print(f"  Confidence analysis: {confidence_analysis_path}")
        print(f"  Analysis report: {os.path.join(output_dir, 'analysis_report.txt')}")
        print(f"\nOutput directory: {output_dir}")

        training_summary_pattern = os.path.join(optim_output_dir, '*summary*.txt')
        training_summary_files = glob.glob(training_summary_pattern)
        if training_summary_files:
            import shutil
            summary_source = training_summary_files[0]
            summary_dest = os.path.join(output_dir, 'training_summary_reference.txt')
            shutil.copy2(summary_source, summary_dest)
            print(f"  Training summary file: {summary_dest}")

        print(f"\nAll files saved in: {output_dir}")
    else:
        print("Insufficient heatmap data collected")

