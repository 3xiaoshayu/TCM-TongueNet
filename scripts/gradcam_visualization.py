import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

import torchvision.models as models
import torch.nn as nn
from datetime import datetime
import glob


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
    def __init__(self, resnet_version='resnet101', pretrained=True, use_attention=False, attention_type='cbam'):
        super(TongueModel5Class, self).__init__()

        import torchvision.models as models
        if pretrained:
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
                if resnet_version in ['resnet18', 'resnet34']:
                    self.attention1 = CBAM(64)
                    self.attention2 = CBAM(128)
                    self.attention3 = CBAM(256)
                    self.attention4 = CBAM(512)
                else:
                    self.attention1 = CBAM(256)
                    self.attention2 = CBAM(512)
                    self.attention3 = CBAM(1024)
                    self.attention4 = CBAM(2048)
            elif attention_type == 'se':
                if resnet_version in ['resnet18', 'resnet34']:
                    self.attention1 = SEAttention(64)
                    self.attention2 = SEAttention(128)
                    self.attention3 = SEAttention(256)
                    self.attention4 = SEAttention(512)
                else:
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


def parse_args():
    parser = argparse.ArgumentParser(description='Tongue Image Grad-CAM Visualization Analysis')
    parser.add_argument('--img_id', type=int, default=15, help='Image ID (e.g., 15)')
    parser.add_argument('--img_suffix', type=str, default='_1.jpg', help='Image suffix (e.g., _1.jpg)')
    parser.add_argument('--target_class', type=str, default='Kidney',
                        help='Target class: None=auto select, or specify: Heart, Lung, Spleen, Liver, Kidney')
    parser.add_argument('--batch_csv', type=str, default=None,
                        help='Batch CSV file path (e.g., test.csv)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Model weight path (default automatically find latest optimized model)')
    parser.add_argument('--data_dir', type=str,
                        default='D:\\tongue_dataset\\HongKongPolyU\\TongueDx2\\tongue-fixed\\datasets',
                        help='Data directory')
    parser.add_argument('--train_output_dir', type=str, default=None,
                        help='Training output directory, automatically load model from it')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default create gradcam_output under training output directory)')
    parser.add_argument('--max_images', type=int, default=30,
                        help='Maximum number of images to process (default: 30)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume processing from where it left off')

    return parser.parse_args()


def find_latest_model(base_dir):
    parent_dir = os.path.dirname(base_dir)
    pattern = os.path.join(parent_dir, "optimized_output_*")
    model_dirs = glob.glob(pattern)

    if not model_dirs:
        pattern = os.path.join(os.getcwd(), "optimized_output_*")
        model_dirs = glob.glob(pattern)

    if not model_dirs:
        pattern = os.path.join(base_dir, "optimized_output_*")
        model_dirs = glob.glob(pattern)

    if not model_dirs:
        print("Warning: No optimized_output_* directory found")
        return None, None

    model_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = model_dirs[0]

    possible_model_files = [
        os.path.join(latest_dir, 'resnet101_5class_best_optimized.pth'),
        os.path.join(latest_dir, 'resnet50_5class_best_optimized.pth'),
        os.path.join(latest_dir, 'best_model.pth'),
        os.path.join(latest_dir, 'model_best.pth')
    ]

    for model_path in possible_model_files:
        if os.path.exists(model_path):
            print(f"Found latest model: {model_path}")
            return model_path, latest_dir

    print(f"Warning: No model file found in {latest_dir}")
    return None, latest_dir


def load_model(model_path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        args_dict = checkpoint.get('args', {})

        resnet_version = args_dict.get('mode', 'full')
        if resnet_version == 'fast':
            resnet_version = 'resnet18'
        elif resnet_version == 'medium':
            resnet_version = 'resnet50'
        else:
            resnet_version = 'resnet101'

        use_attention = False
        attention_type = 'cbam'
        optim_type = args_dict.get('optim', 'baseline')

        if 'attention' in optim_type:
            use_attention = True
            if 'se' in optim_type:
                attention_type = 'se'

        print(f"Model config: ResNet={resnet_version}, attention={use_attention}, type={attention_type}")

        model = TongueModel5Class(
            resnet_version=resnet_version,
            pretrained=False,
            use_attention=use_attention,
            attention_type=attention_type
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, args_dict

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load with simplified method...")

        try:
            model = TongueModel5Class(resnet_version='resnet101', pretrained=False).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            return model, {}
        except:
            print("Simplified method also failed, trying other method...")

            model = torch.load(model_path, map_location=device)
            if isinstance(model, dict) and 'model_state_dict' in model:
                model_dict = model['model_state_dict']
                model = TongueModel5Class(resnet_version='resnet101', pretrained=False).to(device)
                model.load_state_dict(model_dict)
            model.eval()

            return model, {}


def get_target_layer(model):
    return model.base_model.layer4


def create_transform(args_dict=None):
    if args_dict:
        mode = args_dict.get('mode', 'full')
        if mode == 'fast':
            img_size = 128
        else:
            img_size = 224
    else:
        img_size = 224

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform, img_size


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor, original_image


def compute_gradcam(model, input_tensor, target_class, target_layer):
    input_tensor.requires_grad = True
    model.eval()

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

    if features is None or features.grad is None:
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


def get_predicted_labels(output, threshold=0.5):
    probs = torch.sigmoid(output).cpu().detach().numpy()[0]
    predicted = []
    for i, prob in enumerate(probs):
        if i < len(LABEL_NAMES) and prob >= threshold:
            predicted.append((LABEL_NAMES[i], prob))
    predicted.sort(key=lambda x: x[1], reverse=True)
    return predicted, probs


def get_true_labels(csv_file, img_name):
    try:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)

            if 'image_path' in df.columns:
                img_ids = []
                for path in df['image_path']:
                    try:
                        import re
                        match = re.search(r'(\d+)_', str(path))
                        if match:
                            img_ids.append(int(match.group(1)))
                        else:
                            filename = os.path.basename(str(path))
                            match = re.search(r'(\d+)_', filename)
                            if match:
                                img_ids.append(int(match.group(1)))
                            else:
                                img_ids.append(-1)
                    except:
                        img_ids.append(-1)

                current_img_id = int(img_name.split('_')[0])
                if current_img_id in img_ids:
                    idx = img_ids.index(current_img_id)
                    img_row = df.iloc[idx]

                    true_labels = []
                    for label in LABEL_NAMES:
                        if label in img_row.index and img_row[label] == 1:
                            true_labels.append(label)
                    return true_labels

        return []
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []


def apply_colormap(cam, image):
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    cam = np.uint8(255 * cam)
    colormap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image, 0.6, colormap, 0.4, 0)
    return blended


SEGMENTATION_LINES = {
    'tip_ratio': 0.40,
    'root_ratio': 0.70,
    'left_ratio': 0.35,
    'right_ratio': 0.65
}

REGIONS = {
    'Root': {'row_start': 0.0, 'row_end': SEGMENTATION_LINES['tip_ratio'], 'col_start': 0.0, 'col_end': 1.0},
    'Mid-Left': {'row_start': SEGMENTATION_LINES['tip_ratio'], 'row_end': SEGMENTATION_LINES['root_ratio'],
                 'col_start': 0.0, 'col_end': SEGMENTATION_LINES['left_ratio']},
    'Mid-Center': {'row_start': SEGMENTATION_LINES['tip_ratio'], 'row_end': SEGMENTATION_LINES['root_ratio'],
                   'col_start': SEGMENTATION_LINES['left_ratio'], 'col_end': SEGMENTATION_LINES['right_ratio']},
    'Mid-Right': {'row_start': SEGMENTATION_LINES['tip_ratio'], 'row_end': SEGMENTATION_LINES['root_ratio'],
                  'col_start': SEGMENTATION_LINES['right_ratio'], 'col_end': 1.0},
    'Tip': {'row_start': SEGMENTATION_LINES['root_ratio'], 'row_end': 1.0, 'col_start': 0.0, 'col_end': 1.0}
}


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
        if region_cam.size > 0:
            results[region_name] = np.sum(region_cam)
        else:
            results[region_name] = 0.0
    return results


def create_combined_visualization(img_name, original_image, gradcam_overlay, cam,
                                  image_size, pred_label_str, target_class_idx,
                                  true_labels, predicted, probs):
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    h, w = image_size

    fig = plt.figure(figsize=(20, 12))

    fig.suptitle(f'Tongue Image Analysis: {img_name}',
                 fontsize=18, fontweight='bold', y=0.98, fontfamily='Times New Roman')

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    true_label_text = ', '.join(true_labels) if true_labels else 'No Label'
    ax1.set_title(f'(a) Original Image\n(True Label: {true_label_text})',
                  fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'(b) Grad-CAM Heatmap\n(Target: {pred_label_str})',
                  fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 3, 3)
    cam_resized = cv2.resize(cam, (w, h))
    cam_uint8 = np.uint8(255 * cam_resized)
    cam_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    line_thickness = 1
    for region_name, coords in REGIONS.items():
        r_start = int(h * coords['row_start'])
        r_end = int(h * coords['row_end'])
        c_start = int(w * coords['col_start'])
        c_end = int(w * coords['col_end'])

        cv2.rectangle(cam_colored, (c_start, r_start), (c_end, r_end), (255, 255, 255), line_thickness)
        mid_r = (r_start + r_end) // 2
        mid_c = (c_start + c_end) // 2

        if region_name == 'Root':
            text_y = r_start + 20
            text_x = mid_c - 15
        elif region_name == 'Tip':
            text_y = r_end - 10
            text_x = mid_c - 10
        elif region_name == 'Mid-Left':
            text_y = mid_r
            text_x = c_start + 10
        elif region_name == 'Mid-Right':
            text_y = mid_r
            text_x = c_end - 60
        else:
            text_y = mid_r
            text_x = mid_c - 35

        cv2.putText(cam_colored, region_name, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    ax3.imshow(cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB))
    ax3.set_title('(c) Grad-CAM with Region Division',
                  fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 3, 4)
    region_scores = compute_region_attention(cam_resized, image_size)
    regions = list(region_scores.keys())
    scores = list(region_scores.values())

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    bars = ax4.bar(regions, scores, color=colors, edgecolor='black', linewidth=1.2)
    ax4.set_ylabel('Attention Intensity (Mean)', fontsize=11, fontfamily='Times New Roman')
    ax4.set_xlabel('Tongue Region', fontsize=11, fontfamily='Times New Roman')
    ax4.set_title('(d) Regional Attention Quantification',
                  fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax4.set_ylim(0, max(scores) * 1.2 if max(scores) > 0 else 1)
    ax4.set_xticklabels(regions, rotation=0, fontproperties='Times New Roman')

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax4.annotate(f'{score:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontfamily='Times New Roman')

    ax5 = fig.add_subplot(2, 3, 5)
    total = sum(scores) if sum(scores) > 0 else 1
    percentages = [(score / total * 100) if total > 0 else 0 for score in scores]

    def autopct_format(pct):
        return ('%.1f%%' % pct) if pct > 5 else ''

    wedges, texts, autotexts = ax5.pie(scores, labels=regions, colors=colors, autopct=autopct_format,
                                       startangle=90, textprops={'fontsize': 9, 'fontfamily': 'Times New Roman'})
    ax5.set_title('(e) Regional Attention Distribution',
                  fontsize=12, fontweight='bold', fontfamily='Times New Roman')

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontfamily('Times New Roman')

    for text in texts:
        text.set_fontfamily('Times New Roman')
        text.set_fontsize(9)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')

    table_data = []
    headers = ['Organ', 'Probability', 'Confidence', 'Prediction', 'Target Organ']

    max_prob_idx = np.argmax(probs)
    max_prob_organ = LABEL_NAMES[max_prob_idx]

    for i, organ in enumerate(LABEL_NAMES):
        prob = probs[i]
        confidence = 'High' if prob >= 0.7 else 'Medium' if prob >= 0.5 else 'Low'

        if organ in predicted:
            prediction_symbol = '●'
        else:
            prediction_symbol = '○'

        is_max_prob = (organ == max_prob_organ)
        probability_str = f"{prob:.3f} {'(Max)' if is_max_prob else ''}"

        is_target = (organ == pred_label_str)
        target_str = '✓' if is_target else ''

        table_data.append([organ, probability_str, confidence, prediction_symbol, target_str])

    table_data.append(['-' * 10, '-' * 15, '-' * 10, '-' * 10, '-' * 10])
    table_data.append(['Summary:', f'{max_prob_organ}',
                       'High' if probs[max_prob_idx] >= 0.7 else 'Medium',
                       '●', pred_label_str])

    table = ax6.table(cellText=table_data, colLabels=headers, loc='center',
                      cellLoc='center', colWidths=[0.18, 0.22, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)

    for key, cell in table.get_celld().items():
        cell.set_fontsize(9)
        cell.get_text().set_fontfamily('Times New Roman')
        if key[0] == 0:
            cell.get_text().set_weight('bold')
            cell.get_text().set_color('white')

    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4F81BD')

    for i, row in enumerate(table_data[:-2]):
        if row[4] == '✓':
            for j in range(len(headers)):
                table[(i + 1, j)].set_facecolor('#C5D9F1')

    for j in range(len(headers)):
        table[(len(table_data), j)].set_facecolor('#D9D9D9')
        table[(len(table_data), j)].get_text().set_weight('bold')

    ax6.set_title('(f) Model Predictions & Confidence',
                  fontsize=12, fontweight='bold', pad=20, fontfamily='Times New Roman')

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


def process_single_image(args, model, transform, data_dir):
    print(f"\nProcessing image: {args.img_id}{args.img_suffix}")

    seg_dir = os.path.join(data_dir, 'seg')

    test_csv = None
    possible_csv_paths = [
        args.batch_csv if args.batch_csv else None,
        os.path.join(data_dir, 'list_new', 'test.csv'),
        os.path.join(data_dir, 'list', 'test.csv'),
        os.path.join(os.path.dirname(data_dir), 'list_new', 'test.csv'),
        os.path.join(os.path.dirname(data_dir), 'list', 'test.csv')
    ]

    for csv_path in possible_csv_paths:
        if csv_path and os.path.exists(csv_path):
            test_csv = csv_path
            break

    img_name = f"{args.img_id}{args.img_suffix}"

    possible_img_paths = [
        os.path.join(seg_dir, 'test', img_name),
        os.path.join(seg_dir, img_name),
        os.path.join(data_dir, 'test', img_name),
        os.path.join(data_dir, 'segmented', 'test', img_name),
        os.path.join(os.path.dirname(data_dir), 'seg', 'test', img_name),
        os.path.join(os.path.dirname(data_dir), 'seg', img_name)
    ]

    img_path = None
    for path in possible_img_paths:
        if os.path.exists(path):
            img_path = path
            break

    if not img_path:
        print(f"Error: Image does not exist - {img_name}")
        print("Tried the following paths:")
        for path in possible_img_paths:
            print(f"  {path}")
        return False

    print(f"Preprocessing image: {img_path}")
    try:
        input_tensor, original_image = preprocess_image(img_path, transform)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return False

    print("Getting predictions...")
    with torch.no_grad():
        output = model(input_tensor)

    predicted, probs = get_predicted_labels(output)

    true_labels = []
    if test_csv and os.path.exists(test_csv):
        true_labels = get_true_labels(test_csv, img_name)
    else:
        print(f"Warning: CSV file not found")

    if predicted:
        if args.target_class is not None and args.target_class in LABEL_NAMES:
            target_class_idx = LABEL_NAMES.index(args.target_class)
            pred_label_str = args.target_class
        else:
            target_class_idx = LABEL_NAMES.index(predicted[0][0])
            pred_label_str = predicted[0][0]

        print(f"Model prediction: {pred_label_str} (probability: {probs[target_class_idx]:.4f})")
        print(f"True labels: {', '.join(true_labels) if true_labels else 'No labels'}")

        print("Computing Grad-CAM...")
        try:
            target_layer = get_target_layer(model)
            cam = compute_gradcam(model, input_tensor, target_class_idx, target_layer)
            gradcam_overlay = apply_colormap(cam, original_image)
        except Exception as e:
            print(f"Error computing Grad-CAM: {e}")
            return False

        print("Creating visualization...")
        try:
            fig = create_combined_visualization(
                img_name, original_image, gradcam_overlay, cam,
                original_image.shape[:2], pred_label_str, target_class_idx,
                true_labels, [p[0] for p in predicted], probs
            )
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return False

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{img_name.split('.')[0]}_gradcam_{timestamp}.png"
        output_path = os.path.join(args.output_dir, output_filename)

        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
        finally:
            plt.close(fig)

        print("\n" + "=" * 60)
        print(f"Quantitative Analysis Results (Target: {pred_label_str})")
        print("=" * 60)
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        region_scores = compute_region_attention(cam_resized, original_image.shape[:2])
        total = sum(region_scores.values()) if sum(region_scores.values()) > 0 else 1

        print(f"{'Region':<12} {'Attention Intensity':<15} {'Percentage':<10}")
        print("-" * 60)
        for region, score in region_scores.items():
            pct = (score / total * 100) if total > 0 else 0
            print(f"{region:<12} {score:<15.4f} {pct:<10.2f}%")
        print("=" * 60)

        return True
    else:
        print("No labels detected (all probabilities below 0.5)")
        return False


def process_batch_images(args, model, transform, data_dir):
    if not args.batch_csv:
        print("Error: Batch CSV file not specified")
        return

    csv_path = args.batch_csv
    possible_csv_paths = [
        csv_path,
        os.path.join(data_dir, 'list_new', 'test.csv'),
        os.path.join(data_dir, 'list', 'test.csv'),
        os.path.join(os.path.dirname(data_dir), 'list_new', 'test.csv'),
        os.path.join(os.path.dirname(data_dir), 'list', 'test.csv')
    ]

    for path in possible_csv_paths:
        if os.path.exists(path):
            csv_path = path
            break

    if not os.path.exists(csv_path):
        print(f"Error: CSV file does not exist - {csv_path}")
        print("Tried the following paths:")
        for path in possible_csv_paths:
            print(f"  {path}")
        return

    print(f"Starting batch processing of CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    img_ids = []
    if 'image_path' in df.columns:
        for path in df['image_path']:
            try:
                import re
                match = re.search(r'(\d+)_', str(path))
                if match:
                    img_ids.append(int(match.group(1)))
                else:
                    filename = os.path.basename(str(path))
                    match = re.search(r'(\d+)_', filename)
                    if match:
                        img_ids.append(int(match.group(1)))
            except:
                continue

    if not img_ids:
        print("Unable to extract image IDs from CSV")
        return

    img_ids = list(set(img_ids))
    img_ids.sort()
    print(f"Found {len(img_ids)} images to process")

    max_process_count = args.max_images
    print(f"Will process up to {max_process_count} images")

    progress_file = os.path.join(args.output_dir, "batch_progress.txt")
    processed_ids = set()

    if args.resume and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and line.isdigit():
                        processed_ids.add(int(line))
            print(f"Resuming from progress file, already processed {len(processed_ids)} images")
        except Exception as e:
            print(f"Cannot read progress file, starting from scratch: {e}")

    remaining_ids = [img_id for img_id in img_ids if img_id not in processed_ids]

    if len(remaining_ids) == 0:
        print("All images already processed")
        return

    remaining_ids = remaining_ids[:max_process_count]

    success_count = 0
    for i, img_id in enumerate(remaining_ids):
        print(f"\n{'=' * 60}")
        print(f"Progress: {i + 1}/{len(remaining_ids)} (Image ID: {img_id})")
        print(f"{'=' * 60}")

        args.img_id = img_id

        try:
            result = process_single_image(args, model, transform, data_dir)
            if result:
                success_count += 1
                with open(progress_file, 'a') as f:
                    f.write(f"{img_id}\n")
        except KeyboardInterrupt:
            print("\nInterrupted by user! Saved completed results.")
            print("You can resume next time with --resume flag")
            break
        except Exception as e:
            print(f"Error processing image {img_id}: {str(e)}")
            continue

    print(f"\n{'=' * 60}")
    print(f"Batch processing completed!")
    print(f"Successfully processed: {success_count}/{len(remaining_ids)} images")
    print(f"Output directory: {args.output_dir}")
    print(f"{'=' * 60}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Tongue Image Grad-CAM Visualization Analysis")
    print("=" * 60)

    model_path = args.model_path
    train_output_dir = args.train_output_dir

    if not model_path:
        result = find_latest_model(args.data_dir)
        if result:
            model_path, train_output_dir = result
        else:
            print("Error: Model file not found, please specify with --model_path")
            return

    if args.output_dir:
        output_dir = args.output_dir
    elif train_output_dir:
        output_dir = os.path.join(train_output_dir, 'gradcam_output')
    else:
        model_dir = os.path.dirname(model_path)
        output_dir = os.path.join(model_dir, 'gradcam_output')

    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    print(f"\nCurrent configuration:")
    print(f"  Image ID: {args.img_id}")
    print(f"  Image suffix: {args.img_suffix}")
    print(f"  Target class: {args.target_class if args.target_class else 'auto select'}")
    print(f"  Batch CSV: {args.batch_csv if args.batch_csv else 'not specified'}")
    print(f"  Model path: {model_path}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Max images: {args.max_images}")
    print(f"  Resume mode: {'yes' if args.resume else 'no'}")

    print("\nLoading model...")
    try:
        model, args_dict = load_model(model_path, device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    transform, img_size = create_transform(args_dict)
    print(f"Image preprocessing size: {img_size}x{img_size}")

    if args.batch_csv:
        process_batch_images(args, model, transform, args.data_dir)
    else:
        process_single_image(args, model, transform, args.data_dir)

    print("\nAnalysis completed!")

