import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from feature_activation_map import TongueModel13Class, TongueDataset13Class

data_dir = r"D:\tongue_dataset\HongKongPolyU\TongueDx2\tongue13\datasets"
output_dir = r"D:\tongue_dataset\HongKongPolyU\TongueDx2\tongue13\13class_output_20260317_223014_with_act_crack_map"
best_model_path = os.path.join(output_dir, '13class_best.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    cfg = checkpoint['model_config']
    model = TongueModel13Class(
        resnet_version=cfg['resnet_version'],
        pretrained=False,
        use_attention=cfg['use_attention'],
        attention_type=cfg['attention_type'],
        num_classes=13
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    list_dir = os.path.join(data_dir, 'list')
    test_df = pd.read_csv(os.path.join(list_dir, 'test.csv'))
    seg_dir = os.path.join(data_dir, 'seg')
    test_dataset = TongueDataset13Class(test_df, seg_dir, test_transform, preload=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    if isinstance(model, torch.nn.DataParallel):
        target_layers = [model.module.base_model.layer4[-1]]
    else:
        target_layers = [model.base_model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    all_feat_maps = []
    all_cam_maps = []
    all_paths = []
    all_labels = []

    for idx, (image, label, path) in enumerate(test_loader):
        img = image.to(device)
        label = label.to(device)
        path_str = path[0]

        with torch.no_grad():
            _, activation_maps = model(img)
            feat_map = activation_maps.squeeze(0).cpu().numpy()
            all_feat_maps.append(feat_map)
            all_labels.append(label.cpu().numpy().squeeze(0))
            all_paths.append(path_str)

        cam_maps = []
        for j in range(5):
            target = ClassifierOutputTarget(8 + j)
            grayscale_cam = cam(input_tensor=img, targets=[target])
            cam_maps.append(grayscale_cam.squeeze(0))
        cam_map = np.stack(cam_maps, axis=0)
        all_cam_maps.append(cam_map)

        if (idx+1) % 100 == 0:
            print(f"Processed {idx+1} images")

    all_feat_maps = np.array(all_feat_maps)
    all_cam_maps = np.array(all_cam_maps)
    all_labels = np.array(all_labels)

    np.savez(os.path.join(output_dir, 'test_all_maps.npz'),
             feat_maps=all_feat_maps,
             cam_maps=all_cam_maps,
             paths=all_paths,
             labels=all_labels)

    print("Feature activation maps and organ CAMs saved to:", os.path.join(output_dir, 'test_all_maps.npz'))
