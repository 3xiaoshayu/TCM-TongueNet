import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models
import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


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
    def __init__(self, resnet_version='resnet101', pretrained=True, use_attention=True, attention_type='se'):
        super(TongueModel5Class, self).__init__()

        if pretrained:
            import torchvision.models as models
            if resnet_version == 'resnet101':
                weights = models.ResNet101_Weights.IMAGENET1K_V1
            else:
                weights = None
        else:
            weights = None

        import torchvision.models as models
        if resnet_version == 'resnet101':
            self.base_model = models.resnet101(weights=weights)
            num_ftrs = self.base_model.fc.in_features
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}")

        self.use_attention = use_attention
        self.attention_type = attention_type

        if use_attention and attention_type == 'se':
            self.attention1 = SEAttention(256)
            self.attention2 = SEAttention(512)
            self.attention3 = SEAttention(1024)
            self.attention4 = SEAttention(2048)

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

REGION_NAMES = ['Tip', 'Mid-Left', 'Mid-Center', 'Mid-Right', 'Root']

SEGMENTATION_LINES = {
    'tip_start': 0.70,
    'mid_start': 0.40,
    'mid_end': 0.70,
    'root_start': 0.0,
    'root_end': 0.40,
    'left_ratio': 0.35,
    'center_left': 0.35,
    'center_right': 0.65,
    'right_ratio': 0.65
}

REGIONS = {
    'Tip': {
        'row_start': SEGMENTATION_LINES['tip_start'],
        'row_end': 1.0,
        'col_start': 0.0,
        'col_end': 1.0
    },
    'Mid-Left': {
        'row_start': SEGMENTATION_LINES['mid_start'],
        'row_end': SEGMENTATION_LINES['mid_end'],
        'col_start': 0.0,
        'col_end': SEGMENTATION_LINES['center_left']
    },
    'Mid-Center': {
        'row_start': SEGMENTATION_LINES['mid_start'],
        'row_end': SEGMENTATION_LINES['mid_end'],
        'col_start': SEGMENTATION_LINES['center_left'],
        'col_end': SEGMENTATION_LINES['center_right']
    },
    'Mid-Right': {
        'row_start': SEGMENTATION_LINES['mid_start'],
        'row_end': SEGMENTATION_LINES['mid_end'],
        'col_start': SEGMENTATION_LINES['center_right'],
        'col_end': 1.0
    },
    'Root': {
        'row_start': SEGMENTATION_LINES['root_start'],
        'row_end': SEGMENTATION_LINES['root_end'],
        'col_start': 0.0,
        'col_end': 1.0
    }
}

TCM_EXPECTATIONS = {
    'Heart': 'Tip',
    'Lung': 'Tip',
    'Spleen': 'Mid-Center',
    'Liver': 'Mid-Left+Mid-Right',
    'Kidney': 'Root'
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = r'D:\tongue_dataset\HongKongPolyU\TongueDx2\tongue-fixed\datasets'


def find_latest_model():
    data_dir_parent = os.path.dirname(data_dir)

    output_dirs = []
    for item in os.listdir(data_dir_parent):
        if item.startswith('optimized_output_') and os.path.isdir(os.path.join(data_dir_parent, item)):
            output_dirs.append(item)

    if not output_dirs:
        raise FileNotFoundError("Training output model file not found")

    output_dirs.sort(reverse=True)
    latest_dir = os.path.join(data_dir_parent, output_dirs[0])

    model_path = os.path.join(latest_dir, 'resnet101_5class_best_optimized.pth')
    if not os.path.exists(model_path):
        for file in os.listdir(latest_dir):
            if file.endswith('.pth') and 'best' in file:
                model_path = os.path.join(latest_dir, file)
                break
        else:
            raise FileNotFoundError(f"Model file not found in {latest_dir}")

    print(f"Found latest model: {model_path}")
    print(f"From training output: {latest_dir}")
    return model_path, latest_dir


def load_model(model_path):
    model = TongueModel5Class(
        resnet_version='resnet101',
        pretrained=False,
        use_attention=True,
        attention_type='se'
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor, original_image


def compute_standard_gradcam(model, input_tensor, target_class, target_layer):
    input_tensor.requires_grad = True

    features = None

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

    if features is None or features.grad is None:
        return np.zeros((7, 7))

    gradients = features.grad

    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

    cam = torch.zeros(features.shape[2:], device=device, dtype=torch.float32)
    for i in range(weights.shape[1]):
        cam += weights[0, i] * features[0, i]

    cam = F.relu(cam)
    cam_np = cam.detach().cpu().numpy()

    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()

    return cam_np


def get_target_layer(model):
    return model.base_model.layer4


def compute_region_attention(cam, image_size):
    h, w = image_size
    results = {}

    for region_name, coords in REGIONS.items():
        r_start = int(h * coords['row_start'])
        r_end = int(h * coords['row_end'])
        c_start = int(w * coords['col_start'])
        c_end = int(w * coords['col_end'])

        region_cam = cam[r_start:r_end, c_start:c_end]
        region_cam = np.maximum(region_cam, 0)

        if region_cam.size == 0:
            results[region_name] = 0.0
            continue

        flat_cam = region_cam.flatten()
        sorted_cam = np.sort(flat_cam)[::-1]

        top_k = int(len(sorted_cam) * 0.24)
        if top_k < 1:
            top_k = 1

        top_values = sorted_cam[:top_k]
        score = np.mean(top_values)
        results[region_name] = score

    max_score = max(results.values()) if max(results.values()) > 0 else 1
    for region in results:
        results[region] = results[region] / max_score

    return results


def check_tcm_match(attention_scores, organ_name):
    expected = TCM_EXPECTATIONS[organ_name]

    max_score = max(attention_scores.values())
    max_regions = [r for r, s in attention_scores.items() if s == max_score]

    if '+' in expected:
        expected_regions = expected.split('+')
        direct_match = any(r in expected_regions for r in max_regions)

        if direct_match:
            return True, max_regions[0], "direct_match"
        else:
            if organ_name == 'Liver':
                left_score = attention_scores.get('Mid-Left', 0)
                right_score = attention_scores.get('Mid-Right', 0)
                center_score = attention_scores.get('Mid-Center', 0)

                sides_avg = (left_score + right_score) / 2
                if sides_avg > center_score * 0.85:
                    return True, f"Sides dominant (L:{left_score:.2f}, R:{right_score:.2f})", "relative_dominance"

    else:
        if expected in max_regions:
            return True, expected, "direct_match"

    return False, max_regions[0], "no_match"


def find_image_path(csv_path):
    seg_dir = os.path.join(data_dir, 'seg')
    new_list_dir = os.path.join(data_dir, 'list_new')

    if os.path.exists(new_list_dir):
        seg_path = os.path.join(seg_dir, os.path.basename(csv_path))
        if os.path.exists(seg_path):
            return seg_path

    seg_path = os.path.join(seg_dir, csv_path)
    if os.path.exists(seg_path):
        return seg_path

    seg_path_replaced = os.path.join(seg_dir, csv_path.replace('/', '\\'))
    if os.path.exists(seg_path_replaced):
        return seg_path_replaced

    filename = os.path.basename(csv_path)
    for subdir in ['0001_1000', '1001_2000', '2001_3000', '3001_4000', '4001_5000', 'test']:
        seg_path = os.path.join(seg_dir, subdir, filename)
        if os.path.exists(seg_path):
            return seg_path

    return None


def create_radar_chart(aggregated, output_path):
    categories = list(LABEL_NAMES)
    regions = REGION_NAMES
    num_vars = len(regions)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']

    data = []
    raw_data = []

    for category in categories:
        values = [aggregated[category][region]['mean'] for region in regions]
        raw_data.append(values)
        max_val = max(values) if max(values) > 0 else 1
        normalized_values = [v / max_val for v in values]
        normalized_values += normalized_values[:1]
        data.append(normalized_values)

    lines = []
    for idx, (category, values) in enumerate(zip(categories, data)):
        line = ax.plot(angles, values,
                       color=colors[idx],
                       linewidth=2.5,
                       label=category,
                       marker='o',
                       markersize=6)
        lines.append(line[0])

        ax.fill(angles, values,
                facecolor=colors[idx],
                alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(regions, fontsize=10)
    ax.set_ylim(0, 1.1)

    ax.grid(True, linestyle='--', alpha=0.6)

    legend_labels = []
    for idx, category in enumerate(categories):
        raw_vals = raw_data[idx]
        max_region = regions[np.argmax(raw_vals)]
        expected = TCM_EXPECTATIONS[category]

        if '+' in expected:
            expected_regions = expected.split('+')
            match = any(e in max_region for e in expected_regions)
        else:
            match = (max_region == expected)

        if match:
            symbol = "Match"
        else:
            symbol = "NoMatch"

        legend_labels.append(f'{category} ({symbol})')

    ax.legend(legend_labels,
              loc='upper right',
              bbox_to_anchor=(1.3, 1.0),
              fontsize=10,
              frameon=True,
              framealpha=0.9,
              edgecolor='gray')

    plt.title('Radar Chart of Category-Region Attention Patterns\n' +
              'Normalized Attention Scores (by Category Maximum)\n' +
              'Symbols: Match = matches TCM theory, NoMatch = does not match',
              fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Radar chart saved to: {output_path}")


def create_comparison_plot(aggregated, output_path):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    for idx, (ax, category) in enumerate(zip(axes, LABEL_NAMES)):
        regions = REGION_NAMES
        actual_values = [aggregated[category][region]['mean'] for region in regions]

        expected_str = TCM_EXPECTATIONS[category]
        if '+' in expected_str:
            expected_regions = expected_str.split('+')
            expected_values = [1 if any(e in r for e in expected_regions) else 0 for r in regions]
        else:
            expected_values = [1 if r == expected_str else 0 for r in regions]

        if max(actual_values) > 0:
            actual_values_norm = [v / max(actual_values) for v in actual_values]
        else:
            actual_values_norm = actual_values

        x = np.arange(len(regions))
        width = 0.35

        actual_bars = ax.bar(x - width / 2, actual_values_norm, width,
                             label='Actual (Normalized)',
                             color='#2E86AB', alpha=0.9, edgecolor='black', linewidth=1)
        expected_bars = ax.bar(x + width / 2, expected_values, width,
                               label='TCM Expected',
                               color='#A23B72', alpha=0.9, edgecolor='black', linewidth=1)

        ax.set_xlabel('Region', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Attention / Expectation', fontsize=10)

        ax.set_title(f'{category}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        if len(actual_values_norm) > 1:
            correlation = np.corrcoef(actual_values_norm, expected_values)[0, 1]
            ax.text(0.02, 0.98, f'r = {correlation:.3f}',
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        max_idx = np.argmax(actual_values_norm)
        actual_bars[max_idx].set_edgecolor('gold')
        actual_bars[max_idx].set_linewidth(2)

        expected_text = f"Expected: {expected_str}"
        ax.text(0.02, 0.90, expected_text,
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, 1.05),
               ncol=2, fontsize=10, frameon=True)

    plt.suptitle('Comparison of Model Attention vs TCM Theory Expectations',
                 fontsize=14, fontweight='bold', y=1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison plot saved to: {output_path}")


def generate_analysis_report(aggregated, stats, output_path):
    report = []
    report.append("=" * 80)
    report.append("TCM Theory Compliance Analysis Report (No Prior Knowledge)")
    report.append("=" * 80)
    report.append(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Number of analyzed images: {sum(len(stats[label]['Tip']) for label in LABEL_NAMES) // len(REGION_NAMES)}")
    report.append("Note: This analysis did not use any TCM prior knowledge to guide attention computation")
    report.append("")

    report.append("1. Overall Compliance Analysis")
    report.append("-" * 40)

    match_count = 0
    category_details = []

    for category in LABEL_NAMES:
        regions = REGION_NAMES
        actual_values = [aggregated[category][region]['mean'] for region in regions]
        max_region = regions[np.argmax(actual_values)]
        expected = TCM_EXPECTATIONS[category]

        if '+' in expected:
            expected_regions = expected.split('+')
            match = any(e in max_region for e in expected_regions)
            expected_display = " or ".join(expected_regions)
        else:
            match = (max_region == expected)
            expected_display = expected

        if match:
            match_count += 1

        category_details.append({
            'category': category,
            'actual': max_region,
            'expected': expected_display,
            'match': match,
            'attention_values': actual_values,
            'max_value': max(actual_values),
            'mean_value': np.mean(actual_values)
        })

        report.append(f"{category}:")
        report.append(f"  Highest attention region: {max_region} ({max(actual_values):.4f})")
        report.append(f"  TCM expected region: {expected_display}")
        report.append(f"  Compliance: {'[Yes]' if match else '[No]'}")
        report.append(f"  Mean attention: {np.mean(actual_values):.4f}")
        report.append("")

    consistency_rate = match_count / len(LABEL_NAMES) * 100
    report.append(f"Overall compliance rate: {consistency_rate:.1f}% ({match_count}/{len(LABEL_NAMES)})")

    report.append("\n2. Detailed Statistical Analysis")
    report.append("-" * 40)

    for category in LABEL_NAMES:
        report.append(f"\n{category}:")
        for region in REGION_NAMES:
            data = aggregated[category][region]
            report.append(f"  {region}: mean={data['mean']:.4f}, std={data['std']:.4f}, count={data['count']}")

    report.append("\n3. Anomaly Discovery Analysis")
    report.append("-" * 40)

    unmatched_categories = [d for d in category_details if not d['match']]
    if unmatched_categories:
        report.append(f"Found {len(unmatched_categories)} categories that do not comply with TCM theory:")
        for detail in unmatched_categories:
            report.append(f"\n{detail['category']}:")
            report.append(f"  Expected region: {detail['expected']}")
            report.append(f"  Actual highest attention region: {detail['actual']}")

            actual_values = detail['attention_values']
            sorted_indices = np.argsort(actual_values)[::-1]
            top3_regions = [REGION_NAMES[i] for i in sorted_indices[:3]]
            top3_values = [actual_values[i] for i in sorted_indices[:3]]
            report.append(f"  Top-3 attention regions: {', '.join([f'{r}({v:.4f})' for r, v in zip(top3_regions, top3_values)])}")
    else:
        report.append("All categories comply with TCM theory!")

    report.append("\n4. Conclusions and Suggestions")
    report.append("-" * 40)

    if consistency_rate >= 80:
        report.append("Conclusion: The model is highly consistent with TCM theory")
        report.append("  Suggestion: The model can be well applied to TCM tongue diagnosis analysis, with high clinical reference value")
    elif consistency_rate >= 60:
        report.append("Conclusion: The model is generally consistent with TCM theory")
        report.append("  Suggestion: The model can be applied to TCM tongue diagnosis, but attention should be paid to differences in individual organs")
    else:
        report.append("Conclusion: The model has limited consistency with TCM theory")
        report.append("  Suggestion: Further research on the relationship between learned features and TCM theory is needed")

    report.append("\n" + "=" * 80)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"Analysis report saved to: {output_path}")
    return report


def main():
    print("=" * 80)
    print("TCM Theory Verification for Tongue Diagnosis Model - No Prior Knowledge Version")
    print("Companion to debug_train2_stable.py")
    print("=" * 80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(data_dir, f'tcm_analysis_no_prior_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    try:
        model_path, training_output_dir = find_latest_model()
        model = load_model(model_path)
        print(f"Model loaded successfully: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return

    test_csv = os.path.join(data_dir, 'list_new', 'test.csv')
    if not os.path.exists(test_csv):
        test_csv = os.path.join(data_dir, 'list', 'test.csv')

    if not os.path.exists(test_csv):
        print(f"Test CSV file not found: {test_csv}")
        return

    print(f"Using test set: {test_csv}")
    df = pd.read_csv(test_csv)
    print(f"Test set size: {len(df)} samples")

    stats = {label: {region: [] for region in REGION_NAMES} for label in LABEL_NAMES}
    match_stats = {label: {'match': 0, 'total': 0} for label in LABEL_NAMES}

    processed_count = 0
    error_count = 0
    no_label_count = 0
    no_image_count = 0

    print("\nStarting image processing...")
    for idx, row in df.iterrows():
        image_path = row['image_path']
        full_path = find_image_path(image_path)

        if full_path is None:
            no_image_count += 1
            continue

        true_labels = []
        for label in LABEL_NAMES:
            if label in row and row[label] == 1:
                true_labels.append(label)

        if not true_labels:
            no_label_count += 1
            continue

        try:
            input_tensor, original_image = preprocess_image(full_path)
            image_size = original_image.shape[:2]
            target_layer = get_target_layer(model)

            for label in true_labels:
                label_idx = LABEL_NAMES.index(label)

                cam = compute_standard_gradcam(model, input_tensor, label_idx, target_layer)
                cam_resized = cv2.resize(cam, (image_size[1], image_size[0]))

                region_scores = compute_region_attention(cam_resized, image_size)

                is_match, match_detail, match_type = check_tcm_match(region_scores, label)

                for region, score in region_scores.items():
                    stats[label][region].append(score)

                match_stats[label]['total'] += 1
                if is_match:
                    match_stats[label]['match'] += 1

            processed_count += 1

            if processed_count % 100 == 0:
                print(f"  Progress: {processed_count}/{len(df)}")

        except Exception as e:
            error_count += 1
            continue

    print(f"\nProcessing completed:")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Errors: {error_count}")
    print(f"  No label images: {no_label_count}")
    print(f"  Image not found: {no_image_count}")

    has_data = False
    for label in LABEL_NAMES:
        if any(len(stats[label][region]) > 0 for region in REGION_NAMES):
            has_data = True
            break

    if not has_data:
        print("\nWarning: No valid data collected!")
        return

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    aggregated = {}
    for label in LABEL_NAMES:
        aggregated[label] = {}
        for region in REGION_NAMES:
            scores = stats[label][region]
            if scores:
                aggregated[label][region] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'count': len(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
            else:
                aggregated[label][region] = {'mean': 0, 'std': 0, 'count': 0, 'min': 0, 'max': 0}

    print(f"\n{'Category':<10} {'Tip':<10} {'Mid-Left':<10} {'Mid-Center':<10} {'Mid-Right':<10} {'Root':<10} {'TCM Match':<10}")
    print("-" * 80)

    total_matches = 0
    for label in LABEL_NAMES:
        regions = REGION_NAMES
        means = [aggregated[label][r]['mean'] for r in regions]
        max_region = regions[np.argmax(means)]

        sample_attention = {region: aggregated[label][region]['mean'] for region in regions}
        is_match, match_detail, match_type = check_tcm_match(sample_attention, label)

        if is_match:
            total_matches += 1
            match_str = "YES"
        else:
            match_str = "NO"

        print(f"{label:<10} {means[0]:<10.4f} {means[1]:<10.4f} {means[2]:<10.4f} {means[3]:<10.4f} {means[4]:<10.4f} {match_str:<10}")

    print("\n" + "=" * 80)
    print("Generating analysis results")
    print("=" * 80)

    radar_output = os.path.join(output_dir, 'attention_radar_chart.png')
    create_radar_chart(aggregated, radar_output)

    comparison_output = os.path.join(output_dir, 'attention_comparison_plot.png')
    create_comparison_plot(aggregated, comparison_output)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for idx, (ax, label) in enumerate(zip(axes, LABEL_NAMES)):
        regions = REGION_NAMES
        means = [aggregated[label][r]['mean'] for r in regions]
        stds = [aggregated[label][r]['std'] for r in regions]

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        bars = ax.bar(regions, means, yerr=stds, color=colors,
                      edgecolor='black', capsize=5, alpha=0.8, linewidth=1)

        ax.set_ylabel('Mean Attention', fontsize=10)
        ax.set_xlabel('Region', fontsize=10)
        ax.set_title(f'{label}', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        max_idx = means.index(max(means))
        bars[max_idx].set_edgecolor('gold')
        bars[max_idx].set_linewidth(3)

    plt.suptitle('Mean Attention Scores by Category and Region (No Prior Knowledge)',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    bar_output = os.path.join(output_dir, 'attention_bar_chart.png')
    plt.savefig(bar_output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar chart: {os.path.basename(bar_output)}")

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    heatmap_data = np.zeros((len(LABEL_NAMES), len(REGION_NAMES)))

    for i, label in enumerate(LABEL_NAMES):
        row_means = []
        for j, region in enumerate(REGION_NAMES):
            row_means.append(aggregated[label][region]['mean'])

        row_sum = sum(row_means)

        if row_sum > 0:
            normalized_row = [mean / row_sum for mean in row_means]
        else:
            normalized_row = [0] * len(REGION_NAMES)

        heatmap_data[i, :] = normalized_row

    for i in range(len(LABEL_NAMES)):
        row_sum = sum(heatmap_data[i, :])
        if abs(row_sum - 1.0) > 0.001:
            print(f"Warning: Row sum for {LABEL_NAMES[i]} is {row_sum:.6f}, not 1")

    im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax2.set_xticks(range(len(REGION_NAMES)))
    ax2.set_yticks(range(len(LABEL_NAMES)))
    ax2.set_xticklabels(REGION_NAMES, rotation=45, ha='right')
    ax2.set_yticklabels(LABEL_NAMES)

    for i in range(len(LABEL_NAMES)):
        for j in range(len(REGION_NAMES)):
            text = ax2.text(j, i, f'{heatmap_data[i, j]:.3f}',
                            ha='center', va='center',
                            color='white' if heatmap_data[i, j] > np.max(heatmap_data) / 2 else 'black',
                            fontsize=9, fontweight='bold')

    cbar = ax2.figure.colorbar(im, ax=ax2, shrink=0.8)
    cbar.ax.set_ylabel('Attention Proportion (Row Normalized)', rotation=-90, va="bottom", fontsize=10)

    ax2.set_title('Category-Region Attention Heatmap\n(Row Normalized: Sum of Each Row = 1)',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Region', fontsize=11)
    ax2.set_ylabel('Category', fontsize=11)

    plt.tight_layout()
    heatmap_output = os.path.join(output_dir, 'category_region_heatmap.png')
    plt.savefig(heatmap_output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap: {os.path.basename(heatmap_output)}")
    print("   Note: Each row in the heatmap is normalized to sum to 1")

    results_data = []
    for label in LABEL_NAMES:
        regions = REGION_NAMES
        means = [aggregated[label][r]['mean'] for r in regions]
        max_region = regions[np.argmax(means)]
        expected = TCM_EXPECTATIONS[label]

        if '+' in expected:
            expected_regions = expected.split('+')
            match = any(e in max_region for e in expected_regions)
        else:
            match = (max_region == expected)

        match_str = "YES" if match else "NO"

        row_data = [label]
        for r in regions:
            m = aggregated[label][r]['mean']
            row_data.append(f"{m:.4f}")
        row_data.append(match_str)
        results_data.append(row_data)

    results_df = pd.DataFrame(results_data,
                              columns=['Category', 'Tip', 'Mid-Left', 'Mid-Center', 'Mid-Right', 'Root', 'TCM_Match'])
    csv_path = os.path.join(output_dir, 'tcm_verification_stats.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Statistics CSV: {os.path.basename(csv_path)}")

    report_path = os.path.join(output_dir, 'tcm_analysis_report.txt')
    report = generate_analysis_report(aggregated, stats, report_path)

    print("\n" + "=" * 80)
    print("Final Conclusion")
    print("=" * 80)

    print(f"\nAmong {len(LABEL_NAMES)} categories, {total_matches} comply with TCM theory.")
    print(f"Compliance rate: {total_matches / len(LABEL_NAMES) * 100:.1f}%")

    if total_matches == 5:
        print("\nThe model is perfectly consistent with TCM theory!")
        print("  This indicates that without TCM prior guidance, the model")
        print("  autonomously learned the tongue-organ correspondence consistent with TCM theory.")
    elif total_matches >= 4:
        print("\nThe model is highly consistent with TCM theory!")
        print("  The features learned by the model mostly conform to TCM theory.")
    elif total_matches >= 3:
        print("\nThe model is generally consistent with TCM theory.")
        print("  The model has some ability to learn TCM tongue diagnosis theory.")
    else:
        print("\nThe model has limited consistency with TCM theory.")
        print("  Further research on the relationship between learned features and TCM theory may be needed.")

    print(f"\nAll analysis results saved to: {output_dir}")
    print("\n" + "=" * 80)
    print("Analysis completed")
    print("=" * 80)

