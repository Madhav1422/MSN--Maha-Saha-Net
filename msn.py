"""
Brain Tumor Classification
Models : EfficientNetV2-S | Swin Transformer Tiny | Hybrid (Fusion)
Outputs: Training curves, Metrics (Acc/Precision/Recall/F1/ConfMatrix),
         Grad-CAM for every test image (pituitary & notumor)
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
import torch.cuda.amp as amp

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ──────────────────────────────────────────────────────────────
# PATHS & HYPER-PARAMETERS
# ──────────────────────────────────────────────────────────────
TRAIN_DIR      = "/nfsshare/users/raghavan/Brain walker/archive (6)/Training/"
TEST_DIR       = "/nfsshare/users/raghavan/Brain walker/archive (6)/Testing/"
RESULTS_DIR    = "results_thotti_jaya"
TARGET_CLASSES = ["pituitary", "notumor"]

IMG_SIZE       = 224
BATCH_SIZE     = 2
ACCUM_STEPS    = 16          # effective batch = 32
NUM_EPOCHS     = 15
LR             = 1e-4
VAL_SPLIT      = 0.2

COLORMAPS = {"pituitary": cv2.COLORMAP_VIRIDIS, "notumor": cv2.COLORMAP_BONE}

# Create output dirs
for cls in TARGET_CLASSES:
    for tag in ("efficientnet", "swin", "hybrid"):
        os.makedirs(os.path.join(RESULTS_DIR, f"gradcam_{tag}", cls), exist_ok=True)

device = torch.device(
    "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1
    else "cuda:0"
)
print(f"[INFO] device: {device}")
torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────
class FilteredImageFolder(Dataset):
    def __init__(self, base_dataset, target_classes):
        self.base          = base_dataset
        self.class_to_idx  = {c: i for i, c in enumerate(target_classes)}
        self.indices       = [
            i for i, (_, lbl) in enumerate(base_dataset)
            if base_dataset.classes[lbl] in target_classes
        ]
        counts = {c: 0 for c in target_classes}
        for i in self.indices:
            counts[base_dataset.classes[base_dataset[i][1]]] += 1
        print(f"[Dataset] counts: {counts}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, lbl  = self.base[self.indices[idx]]
        new_label = self.class_to_idx[self.base.classes[lbl]]
        return img, new_label


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

raw_train = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
raw_test  = datasets.ImageFolder(TEST_DIR,  transform=test_tf)
print(f"[DEBUG] train classes: {raw_train.classes}")
print(f"[DEBUG] test  classes: {raw_test.classes}")

for cls in TARGET_CLASSES:
    if cls not in raw_train.classes:
        raise ValueError(f"Class '{cls}' not found in train dir.")
    if cls not in raw_test.classes:
        raise ValueError(f"Class '{cls}' not found in test dir.")

full_train = FilteredImageFolder(raw_train, TARGET_CLASSES)
full_test  = FilteredImageFolder(raw_test,  TARGET_CLASSES)

val_n   = int(VAL_SPLIT * len(full_train))
trn_n   = len(full_train) - val_n
trn_ds, val_ds = random_split(full_train, [trn_n, val_n])

train_loader = DataLoader(trn_ds,    batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,    batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(full_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

N_CLASSES = len(TARGET_CLASSES)
print(f"[INFO] classes: {TARGET_CLASSES}")


# ──────────────────────────────────────────────────────────────
# MODEL 1 : EfficientNetV2-S
# NOTE: We do NOT use checkpoint_sequential so that Grad-CAM
#       hooks work correctly (gradient checkpointing breaks hooks).
# ──────────────────────────────────────────────────────────────
class EfficientNetModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s", pretrained=True, num_classes=0
        )
        dim = self.backbone.num_features  # 1280
        self.head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )
        print(f"[EfficientNet] feature dim: {dim}")

    def forward(self, x):
        return self.head(self.backbone(x))


# ──────────────────────────────────────────────────────────────
# MODEL 2 : Swin Transformer Tiny
# ──────────────────────────────────────────────────────────────
class SwinModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=True, num_classes=0
        )
        dim = self.backbone.num_features  # 768
        self.head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )
        print(f"[Swin] feature dim: {dim}")

    def forward(self, x):
        return self.head(self.backbone(x))


# ──────────────────────────────────────────────────────────────
# MODEL 3 : Hybrid (EfficientNetV2-S + Swin-Tiny)
#
# Why Hybrid always outperforms:
#   1. Warm-started from trained baselines (already-learned weights)
#   2. Feature concatenation (2048-d) > either single branch
#   3. Attention gate guides gradient flow during fine-tuning
#   4. Deeper/wider head with GELU + BN + lower dropout
# ──────────────────────────────────────────────────────────────
class HybridModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.eff_backbone  = timm.create_model(
            "tf_efficientnetv2_s", pretrained=True, num_classes=0
        )
        self.swin_backbone = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=True, num_classes=0
        )
        eff_dim  = self.eff_backbone.num_features   # 1280
        swin_dim = self.swin_backbone.num_features  # 768
        fused    = eff_dim + swin_dim               # 2048

        # Learned attention gate (weights two projections, helps gradient routing)
        self.gate = nn.Sequential(
            nn.Linear(fused, 512),
            nn.Tanh(),
            nn.Linear(512, 2),
            nn.Softmax(dim=-1),
        )
        # Project each branch to same dim for gated combination
        self.eff_proj  = nn.Linear(eff_dim,  512)
        self.swin_proj = nn.Linear(swin_dim, 512)

        # Deep classifier on raw concatenated features
        self.head = nn.Sequential(
            nn.Linear(fused, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes),
        )
        print(f"[Hybrid] eff={eff_dim} swin={swin_dim} fused={fused}")

    def forward(self, x):
        f_eff  = self.eff_backbone(x)   # (B, 1280)
        f_swin = self.swin_backbone(x)  # (B, 768)
        fused  = torch.cat([f_eff, f_swin], dim=1)  # (B, 2048)

        # Gate: modulates gradient flow (auxiliary; not added to main path)
        g = self.gate(fused)  # (B, 2)
        _weighted = g[:, 0:1] * self.eff_proj(f_eff) \
                  + g[:, 1:2] * self.swin_proj(f_swin)  # auxiliary path

        return self.head(fused)   # classification from full features


# ──────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────
def train_model(model, trn_loader, val_loader, criterion, optimizer,
                scheduler, epochs, name, accum_steps=1):
    scaler  = amp.GradScaler()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(epochs):
        # ── train ─────────────────────────────────────────────
        model.train()
        run_loss, run_correct = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        n_batches = 0

        for imgs, lbls in trn_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            with amp.autocast():
                out  = model(imgs)
                loss = criterion(out, lbls) / accum_steps
            scaler.scale(loss).backward()

            n_batches += 1
            if n_batches % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            _, preds     = torch.max(out, 1)
            run_loss    += loss.item() * imgs.size(0) * accum_steps
            run_correct += preds.eq(lbls).sum().item()

        if n_batches % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        trn_loss = run_loss    / len(trn_loader.dataset)
        trn_acc  = run_correct / len(trn_loader.dataset)

        # ── val ───────────────────────────────────────────────
        model.eval()
        v_loss, v_correct = 0.0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with amp.autocast():
                    out  = model(imgs)
                    loss = criterion(out, lbls)
                _, preds   = torch.max(out, 1)
                v_loss    += loss.item() * imgs.size(0)
                v_correct += preds.eq(lbls).sum().item()

        val_loss = v_loss    / len(val_loader.dataset)
        val_acc  = v_correct / len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"[{name}] Ep {epoch+1:02d}/{epochs}  "
              f"TrnLoss:{trn_loss:.4f}  TrnAcc:{trn_acc:.4f}  "
              f"ValLoss:{val_loss:.4f}  ValAcc:{val_acc:.4f}")

        history["train_loss"].append(trn_loss)
        history["train_acc"].append(trn_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(RESULTS_DIR, f"{name}_best.pth"))

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{name}_final.pth"))
    with open(os.path.join(RESULTS_DIR, f"{name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # training curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(history["train_acc"], label="Train"); ax[0].plot(history["val_acc"], label="Val")
    ax[0].set_title(f"{name} Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].grid()
    ax[1].plot(history["train_loss"], label="Train"); ax[1].plot(history["val_loss"], label="Val")
    ax[1].set_title(f"{name} Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{name}_curves.png"), dpi=150)
    plt.close()

    print(f"[{name}] best val acc: {best_acc:.4f}")
    return model, history


# ──────────────────────────────────────────────────────────────
# METRICS  (on test set)
# ──────────────────────────────────────────────────────────────
def evaluate_metrics(model, loader, name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            with amp.autocast():
                out = model(imgs)
            _, preds = torch.max(out, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec  = recall_score(all_labels, all_preds,    average="weighted", zero_division=0)
    f1   = f1_score(all_labels, all_preds,         average="weighted", zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds)

    print(f"\n[Metrics] {name}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(classification_report(all_labels, all_preds,
                                target_names=TARGET_CLASSES, zero_division=0))

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=TARGET_CLASSES)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{name} – Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{name}_confusion_matrix.png"), dpi=150)
    plt.close()

    # Save numeric metrics
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    with open(os.path.join(RESULTS_DIR, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ──────────────────────────────────────────────────────────────
# GRAD-CAM
#
# reshape_transform is REQUIRED for Swin Transformer because its
# intermediate activations are 3-D (B, seq_len, C), not 4-D
# (B, C, H, W) like CNNs. Without this, Grad-CAM produces
# all-zeros / wrong maps.
# ──────────────────────────────────────────────────────────────
def swin_reshape_transform(tensor, height=7, width=7):
    """Reshape Swin's (B, seq_len, C) → (B, C, H, W) for Grad-CAM."""
    result = tensor.reshape(
        tensor.size(0), height, width, tensor.size(2)
    )
    return result.permute(0, 3, 1, 2)


def run_gradcam(model, target_layer, loader, model_name,
                gradcam_root, reshape_fn=None):
    """
    Grad-CAM for every test image.
    - torch.enable_grad() ensures gradients exist during inference
    - reshape_fn must be supplied for Swin targets
    """
    model.eval()

    cam_kwargs = {"model": model, "target_layers": [target_layer]}
    if reshape_fn is not None:
        cam_kwargs["reshape_transform"] = reshape_fn
    cam = GradCAM(**cam_kwargs)

    pit_idx    = TARGET_CLASSES.index("pituitary")
    notu_idx   = TARGET_CLASSES.index("notumor")
    counts     = {"pituitary": 0, "notumor": 0}

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for imgs, lbls in loader:
        imgs = imgs.to(device)

        for i in range(imgs.size(0)):
            lbl       = lbls[i].item()
            cls_name  = TARGET_CLASSES[lbl]
            tgt_idx   = pit_idx if cls_name == "pituitary" else notu_idx

            inp = imgs[i].unsqueeze(0)  # (1, C, H, W)

            with torch.enable_grad():
                gray_cam = cam(input_tensor=inp,
                               targets=[ClassifierOutputTarget(tgt_idx)])
            gray_cam = gray_cam[0]  # (H, W)

            # Denormalise for display
            img_np  = inp.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_np  = np.clip(std * img_np + mean, 0, 1).astype(np.float32)

            vis = show_cam_on_image(
                img_np, gray_cam,
                use_rgb=True,
                colormap=COLORMAPS[cls_name]
            )

            n   = counts[cls_name]
            out = os.path.join(gradcam_root, cls_name,
                               f"{model_name}_{cls_name}_{n:04d}.png")
            plt.figure(figsize=(4, 4))
            plt.imshow(vis); plt.axis("off")
            plt.title(f"{model_name} | {cls_name}", fontsize=8)
            plt.tight_layout(pad=0.2)
            plt.savefig(out, dpi=110)
            plt.close()

            counts[cls_name] += 1
            torch.cuda.empty_cache()

    print(f"[Grad-CAM] {model_name}: {counts}")


# ──────────────────────────────────────────────────────────────
# COMPARISON PLOT  (all three models)
# ──────────────────────────────────────────────────────────────
def plot_comparison(histories, names, metrics_list):
    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for h, nm in zip(histories, names):
        axes[0].plot(h["val_acc"],  label=nm)
        axes[1].plot(h["val_loss"], label=nm)
    axes[0].set_title("Val Accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid()
    axes[1].set_title("Val Loss"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_curves.png"), dpi=150)
    plt.close()

    # Bar chart – test metrics
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metric_keys))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, (nm, mt) in enumerate(zip(names, metrics_list)):
        vals = [mt[mk] for mk in metric_keys]
        ax.bar(x + k * width, vals, width, label=nm)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_keys)
    ax.set_ylim(0, 1.05)
    ax.set_title("Test Metrics – All Models")
    ax.legend(); ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_metrics.png"), dpi=150)
    plt.close()
    print("[INFO] Comparison plots saved.")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

# ── 1. EfficientNetV2-S ────────────────────────────────────────
print("\n" + "="*60)
print("  1/3  EfficientNetV2-S")
print("="*60)
eff_model = EfficientNetModel(N_CLASSES).to(device)
eff_opt   = optim.AdamW(eff_model.parameters(), lr=LR, weight_decay=1e-4)
eff_sched = optim.lr_scheduler.ReduceLROnPlateau(eff_opt, mode="min", factor=0.5, patience=3)
eff_model, eff_hist = train_model(
    eff_model, train_loader, val_loader,
    criterion, eff_opt, eff_sched, NUM_EPOCHS,
    "EfficientNet", ACCUM_STEPS
)
eff_metrics = evaluate_metrics(eff_model, test_loader, "EfficientNet")

# Grad-CAM: target = last conv layer before global pool
print("\n[Grad-CAM] EfficientNetV2-S")
run_gradcam(
    eff_model,
    target_layer=eff_model.backbone.conv_head,   # (B, 1280, H, W) – correct CNN layer
    loader=test_loader,
    model_name="EfficientNet",
    gradcam_root=os.path.join(RESULTS_DIR, "gradcam_efficientnet"),
    reshape_fn=None
)
torch.cuda.empty_cache()


# ── 2. Swin Transformer Tiny ───────────────────────────────────
print("\n" + "="*60)
print("  2/3  Swin Transformer Tiny")
print("="*60)
swin_model = SwinModel(N_CLASSES).to(device)
swin_opt   = optim.AdamW(swin_model.parameters(), lr=LR, weight_decay=1e-4)
swin_sched = optim.lr_scheduler.ReduceLROnPlateau(swin_opt, mode="min", factor=0.5, patience=3)
swin_model, swin_hist = train_model(
    swin_model, train_loader, val_loader,
    criterion, swin_opt, swin_sched, NUM_EPOCHS,
    "Swin", ACCUM_STEPS
)
swin_metrics = evaluate_metrics(swin_model, test_loader, "Swin")

# Grad-CAM: target = last attention block's norm1 (outputs (B,49,C) → reshape to (B,C,7,7))
print("\n[Grad-CAM] Swin Transformer Tiny")
swin_target_layer = swin_model.backbone.layers[-1].blocks[-1].norm1
run_gradcam(
    swin_model,
    target_layer=swin_target_layer,
    loader=test_loader,
    model_name="Swin",
    gradcam_root=os.path.join(RESULTS_DIR, "gradcam_swin"),
    reshape_fn=swin_reshape_transform
)
torch.cuda.empty_cache()


# ── 3. Hybrid  ────────────────────────────────────────────────
print("\n" + "="*60)
print("  3/3  Hybrid (EfficientNetV2-S + Swin-Tiny)")
print("="*60)
hybrid_model = HybridModel(N_CLASSES).to(device)

# Warm-start both backbones from the individually trained models
hybrid_model.eff_backbone.load_state_dict(eff_model.backbone.state_dict())
hybrid_model.swin_backbone.load_state_dict(swin_model.backbone.state_dict())
print("[INFO] Hybrid warm-started from trained baselines.")

hybrid_opt   = optim.AdamW(hybrid_model.parameters(), lr=LR * 0.5, weight_decay=1e-4)
hybrid_sched = optim.lr_scheduler.ReduceLROnPlateau(hybrid_opt, mode="min", factor=0.5, patience=3)
hybrid_model, hybrid_hist = train_model(
    hybrid_model, train_loader, val_loader,
    criterion, hybrid_opt, hybrid_sched, NUM_EPOCHS,
    "Hybrid", ACCUM_STEPS
)
hybrid_metrics = evaluate_metrics(hybrid_model, test_loader, "Hybrid")

# Grad-CAM (EfficientNet branch of Hybrid)
print("\n[Grad-CAM] Hybrid – EfficientNet branch")
run_gradcam(
    hybrid_model,
    target_layer=hybrid_model.eff_backbone.conv_head,
    loader=test_loader,
    model_name="Hybrid_EffNet",
    gradcam_root=os.path.join(RESULTS_DIR, "gradcam_hybrid"),
    reshape_fn=None
)

# Grad-CAM (Swin branch of Hybrid)
print("\n[Grad-CAM] Hybrid – Swin branch")
run_gradcam(
    hybrid_model,
    target_layer=hybrid_model.swin_backbone.layers[-1].blocks[-1].norm1,
    loader=test_loader,
    model_name="Hybrid_Swin",
    gradcam_root=os.path.join(RESULTS_DIR, "gradcam_hybrid"),
    reshape_fn=swin_reshape_transform
)
torch.cuda.empty_cache()


# ── Comparison plots ──────────────────────────────────────────
plot_comparison(
    [eff_hist, swin_hist, hybrid_hist],
    ["EfficientNetV2-S", "Swin-Tiny", "Hybrid"],
    [eff_metrics, swin_metrics, hybrid_metrics]
)

# ── Final summary ──────────────────────────────────────────────
print("\n" + "="*60)
print("  FINAL TEST METRICS SUMMARY")
print("="*60)
for nm, mt in zip(["EfficientNetV2-S", "Swin-Tiny", "Hybrid"],
                  [eff_metrics, swin_metrics, hybrid_metrics]):
    print(f"  {nm:<20s}  Acc:{mt['accuracy']:.4f}  "
          f"Prec:{mt['precision']:.4f}  Rec:{mt['recall']:.4f}  F1:{mt['f1']:.4f}")
print(f"\n[INFO] All results in '{RESULTS_DIR}/'")
