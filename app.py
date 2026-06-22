import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# Model definition (must match train.py exactly)
# ──────────────────────────────────────────────────────────────
IMG_SIZE = 256


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, dropout=dropout))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x  = self.up(x)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x  = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1       = DoubleConv(in_channels, 64)
        self.enc2       = Down(64,  128)
        self.enc3       = Down(128, 256)
        self.enc4       = Down(256, 512)
        self.bottleneck = Down(512, 1024, dropout=0.2)
        self.dec4       = Up(1024 + 512, 512)
        self.dec3       = Up( 512 + 256, 256)
        self.dec2       = Up( 256 + 128, 128)
        self.dec1       = Up( 128 +  64,  64)
        self.out_conv   = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b  = self.bottleneck(s4)
        x  = self.dec4(b,  s4)
        x  = self.dec3(x,  s3)
        x  = self.dec2(x,  s2)
        x  = self.dec1(x,  s1)
        return self.out_conv(x)  # logits


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(ckpt_path: str):
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model     = UNet()
    state     = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    threshold = ckpt.get("threshold", 0.5) if isinstance(ckpt, dict) else 0.5
    metrics   = ckpt.get("test_metrics", {}) if isinstance(ckpt, dict) else {}
    return model, float(threshold), metrics


_img_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess(img: Image.Image) -> torch.Tensor:
    return _img_transform(img.convert("RGB")).unsqueeze(0)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.squeeze() * _std + _mean).permute(1, 2, 0).clamp(0, 1).numpy()


def run_inference(model, img_tensor: torch.Tensor, threshold: float):
    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.sigmoid(logits).squeeze().numpy()
    return probs, (probs > threshold).astype(np.float32)


def pixel_metrics(pred: np.ndarray, gt: np.ndarray):
    eps = 1e-8
    TP  = float(((pred == 1) & (gt == 1)).sum())
    FP  = float(((pred == 1) & (gt == 0)).sum())
    FN  = float(((pred == 0) & (gt == 1)).sum())
    TN  = float(((pred == 0) & (gt == 0)).sum())
    return {
        "Dice / F1":   2 * TP / (2 * TP + FP + FN + eps),
        "IoU":         TP / (TP + FP + FN + eps),
        "Precision":   TP / (TP + FP + eps),
        "Recall":      TP / (TP + FN + eps),
        "Accuracy":    (TP + TN) / (TP + TN + FP + FN + eps),
    }


# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Brain Tumor Segmentation")
st.markdown(
    "Upload an MRI scan — the U-Net model will segment tumor regions pixel by pixel."
)

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model")
    ckpt_path = st.text_input("Checkpoint file", "unet_brain_tumor_best.pth")

    # --- АВТОМАТИЧЕСКОЕ СКАЧИВАНИЕ С GOOGLE ДИСКА ДЛЯ СТРИМЛИТ КЛАУД ---
    import os
    import urllib.request

    if ckpt_path == "unet_brain_tumor_best.pth" and not os.path.exists(ckpt_path):
        with st.spinner("Downloading model weights from Google Drive (~120 MB)... Please wait."):
            # Превращаем твою ссылку в прямую ссылку на скачивание:
            gdrive_id = "1lNU1SafiT8nmEDJbMKiqrn1pLLwpw36W"
            direct_url = f"https://docs.google.com/uc?export=download&id={gdrive_id}"
            
            # Скачиваем файл
            urllib.request.urlretrieve(direct_url, ckpt_path)
            st.success("Weights downloaded successfully!")
    # ─────────────────────────────────────────────────────────────────

    try:
        model, saved_threshold, saved_metrics = load_model(ckpt_path)
        st.success("Model loaded")

        if saved_metrics:
            st.markdown("**Saved test metrics**")
            cols = st.columns(2)
            cols[0].metric("Dice", f"{saved_metrics.get('f1', 0):.3f}")
            cols[1].metric("IoU",  f"{saved_metrics.get('iou', 0):.3f}")
            cols[0].metric("Precision", f"{saved_metrics.get('precision', 0):.3f}")
            cols[1].metric("Recall",    f"{saved_metrics.get('recall', 0):.3f}")

    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    st.divider()
    st.header("Settings")
    threshold = st.slider(
        "Detection threshold",
        min_value=0.05, max_value=0.95,
        value=float(saved_threshold), step=0.05,
        help="Lower = more sensitive. Higher = more precise.",
    )
    show_heatmap = st.checkbox("Show probability heatmap", value=True)

# ──────────────────────────────────────────────────────────────
# Upload
# ──────────────────────────────────────────────────────────────
col_up1, col_up2 = st.columns(2)
with col_up1:
    mri_file = st.file_uploader(
        "MRI scan (required)",
        type=["tif", "tiff", "jpg", "jpeg", "png"],
    )
with col_up2:
    gt_file = st.file_uploader(
        "Ground-truth mask (optional — enables metrics)",
        type=["tif", "tiff", "jpg", "jpeg", "png"],
    )

if not mri_file:
    st.info("Upload an MRI image to start.")
    st.stop()

# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────
img        = Image.open(mri_file)
img_tensor = preprocess(img)
probs, pred_mask = run_inference(model, img_tensor, threshold)
img_disp   = denormalize(img_tensor)

tumor_px  = int(pred_mask.sum())
tumor_pct = 100.0 * tumor_px / (IMG_SIZE * IMG_SIZE)

# Result banner
if tumor_px > 0:
    st.error(
        f"**Tumor detected** — {tumor_pct:.1f}% of image area  "
        f"({tumor_px:,} pixels out of {IMG_SIZE*IMG_SIZE:,})"
    )
else:
    st.success("**No tumor detected** at the current threshold.")

st.divider()

# ──────────────────────────────────────────────────────────────
# Load + resize ground-truth mask if provided
# ──────────────────────────────────────────────────────────────
gt_mask = None
if gt_file:
    gt_pil  = Image.open(gt_file).convert("L").resize(
        (IMG_SIZE, IMG_SIZE), Image.NEAREST
    )
    gt_mask = (np.array(gt_pil) > 127).astype(np.float32)

# ──────────────────────────────────────────────────────────────
# Display results
# ──────────────────────────────────────────────────────────────
n_cols = 2 + int(show_heatmap) + int(gt_mask is not None)
cols   = st.columns(n_cols)
idx    = 0

with cols[idx]:
    st.subheader("MRI Scan")
    st.image(img_disp, use_column_width=True)
idx += 1

with cols[idx]:
    st.subheader("Predicted Mask")
    st.image(pred_mask, use_column_width=True, clamp=True)
idx += 1

if show_heatmap:
    with cols[idx]:
        st.subheader("Probability Map")
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(probs, cmap="hot", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    idx += 1

if gt_mask is not None:
    with cols[idx]:
        st.subheader("Overlay vs. GT")
        overlay = img_disp.copy()
        overlay[(pred_mask == 1) & (gt_mask == 1)] = [0.0, 1.0, 0.0]  # TP green
        overlay[(pred_mask == 1) & (gt_mask == 0)] = [1.0, 0.0, 0.0]  # FP red
        overlay[(pred_mask == 0) & (gt_mask == 1)] = [0.0, 0.0, 1.0]  # FN blue
        st.image(overlay, use_column_width=True)
        st.caption("Green = TP  |  Red = FP  |  Blue = FN")
    idx += 1

# ──────────────────────────────────────────────────────────────
# Metrics (only when GT is provided)
# ──────────────────────────────────────────────────────────────
if gt_mask is not None:
    st.divider()
    st.subheader("Pixel-level Metrics vs. Ground Truth")
    m = pixel_metrics(pred_mask, gt_mask)
    metric_cols = st.columns(len(m))
    for col, (name, val) in zip(metric_cols, m.items()):
        col.metric(name, f"{val:.4f}")
