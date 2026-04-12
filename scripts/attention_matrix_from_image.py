
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

data_dir = r"D:\tongue_dataset\HongKongPolyU\TongueDx2\tongue13\datasets"

def load_all_labels(data_dir):
    list_dir = os.path.join(data_dir, 'list')
    train_df = pd.read_csv(os.path.join(list_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(list_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(list_dir, 'test.csv'))
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    feature_cols = ['TonguePale','TipSideRed','Spot','Ecchymosis','Crack','Toothmark','FurThick','FurYellow']
    organ_cols = ['Heart','Lung','Spleen','Liver','Kidney']
    X = df[feature_cols].values.astype(np.float32)
    y = df[organ_cols].values.astype(np.float32)
    return X, y

print("Loading data...")
X, y = load_all_labels(data_dir)
print(f"Total samples: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

X_train_t = torch.tensor(X_train).to(device)
y_train_t = torch.tensor(y_train).to(device)
X_test_t = torch.tensor(X_test).to(device)
y_test_t = torch.tensor(y_test).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class MLPWithLinearSkip(nn.Module):
    def __init__(self, input_dim=8, hidden1=64, hidden2=32, output_dim=5, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_out = nn.Linear(hidden2, output_dim)
        self.skip = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc_out(out)
        skip_out = self.skip(x)
        return out + skip_out

model = MLPWithLinearSkip().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting training MLP model (with linear bypass)...")
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 20 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    probs = torch.sigmoid(logits).cpu().numpy()

organ_names = ['Heart','Lung','Spleen','Liver','Kidney']
auc_scores = []
for j in range(5):
    auc = roc_auc_score(y_test[:, j], probs[:, j])
    auc_scores.append(auc)
    print(f"{organ_names[j]} AUC: {auc:.3f}")
print(f"Average AUC: {np.mean(auc_scores):.3f}")

skip_weight = model.skip.weight.detach().cpu().numpy()
W = skip_weight.T
print(f"Attention matrix shape: {W.shape}")

feature_names = ['TonguePale','TipSideRed','Spot','Ecchymosis','Crack','Toothmark','FurThick','FurYellow']
crack_idx = feature_names.index('Crack')
spleen_idx = organ_names.index('Spleen')
kidney_idx = organ_names.index('Kidney')

print("\n========== Attention Matrix (rows=features, columns=organs) ==========")
print("        ", "  ".join([f"{org:6s}" for org in organ_names]))
for i, feat in enumerate(feature_names):
    row = "  ".join([f"{W[i,j]:6.3f}" for j in range(5)])
    print(f"{feat:12s} {row}")

print(f"\nCrack->Spleen weight: {W[crack_idx, spleen_idx]:.3f}")
print(f"Crack->Kidney weight: {W[crack_idx, kidney_idx]:.3f}")

output_npy = os.path.join(data_dir, 'attention_W_mlp.npy')
np.save(output_npy, W)
print(f"\nAttention matrix saved to: {output_npy}")

plt.figure(figsize=(12, 6))
sns.heatmap(W, annot=True, fmt='.3f', cmap='coolwarm',
            xticklabels=organ_names, yticklabels=feature_names,
            cbar_kws={'label': 'Weight Value'})
plt.title('Attention Matrix (MLP with Linear Bypass)')
plt.tight_layout()
heatmap_path = os.path.join(data_dir, 'attention_heatmap_mlp.png')
plt.savefig(heatmap_path, dpi=300)
print(f"Heatmap saved to: {heatmap_path}")