
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

data_dir = r"D:\tongue_dataset\HongKongPolyU\TongueDx2\tongue13\datasets"
image_output_dir = r"D:\tongue_dataset\HongKongPolyU\TongueDx2\tongue13\13class_output_20260317_223014_with_act_crack_map"

csv_matrix_path = os.path.join(data_dir, "attention_W_mlp.npy")
img_matrix_path = os.path.join(image_output_dir, "global_attention_from_images.npy")

out_prefix = os.path.join(image_output_dir, "fusion_mlp")

print("Loading matrices...")
W_csv_raw = np.load(csv_matrix_path)
W_img = np.load(img_matrix_path)

print(f"CSV matrix range: [{W_csv_raw.min():.3f}, {W_csv_raw.max():.3f}]")
print(f"Image matrix range: [{W_img.min():.3f}, {W_img.max():.3f}]")

W_csv = (W_csv_raw - W_csv_raw.min()) / (W_csv_raw.max() - W_csv_raw.min() + 1e-8)

alpha = 0.5
M_fusion = alpha * W_img + (1 - alpha) * W_csv
print(f"Fused matrix range: [{M_fusion.min():.3f}, {M_fusion.max():.3f}]")

feature_names = ['TonguePale','TipSideRed','Spot','Ecchymosis','Crack',
                 'Toothmark','FurThick','FurYellow']
organ_names = ['Heart','Lung','Spleen','Liver','Kidney']

plt.figure(figsize=(10, 6))
sns.heatmap(M_fusion, annot=True, fmt='.3f', cmap='viridis',
            xticklabels=organ_names, yticklabels=feature_names,
            cbar_kws={'label': 'Fusion Weight Value'})
plt.title(f'Fusion Matrix (MLP, α={alpha})')
plt.tight_layout()
plt.savefig(out_prefix + ".png", dpi=300)
print(f"Heatmap saved: {out_prefix}.png")

df_fusion = pd.DataFrame(M_fusion, index=feature_names, columns=organ_names)
df_fusion.to_csv(out_prefix + ".csv")
print(f"Fusion matrix saved: {out_prefix}.csv")

print("Done!")