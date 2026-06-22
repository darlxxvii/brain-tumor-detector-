import os

import requests
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# U-Net definition (must match train.py exactly)
# ──────────────────────────────────────────────────────────────
IMG_SIZE = 256
CKPT_PATH = "unet_brain_tumor_best.pth"


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
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
        self.block = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, dropout=dropout)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1       = DoubleConv(3, 64)
        self.enc2       = Down(64, 128)
        self.enc3       = Down(128, 256)
        self.enc4       = Down(256, 512)
        self.bottleneck = Down(512, 1024, dropout=0.2)
        self.dec4       = Up(1024 + 512, 512)
        self.dec3       = Up(512 + 256, 256)
        self.dec2       = Up(256 + 128, 128)
        self.dec1       = Up(128 + 64, 64)
        self.out_conv   = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b  = self.bottleneck(s4)
        x  = self.dec4(b, s4)
        x  = self.dec3(x, s3)
        x  = self.dec2(x, s2)
        x  = self.dec1(x, s1)
        return self.out_conv(x)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(CKPT_PATH):
        return None, 0.5, {}

    ckpt      = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    net       = UNet()
    state     = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
    net.load_state_dict(state)
    net.eval()
    threshold = float(ckpt.get("threshold", 0.5)) if isinstance(ckpt, dict) else 0.5
    metrics   = ckpt.get("test_metrics", {})       if isinstance(ckpt, dict) else {}
    return net, threshold, metrics


_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess(img: Image.Image) -> torch.Tensor:
    return _transform(img.convert("RGB")).unsqueeze(0)


def denormalize(t: torch.Tensor) -> np.ndarray:
    return (t.squeeze() * _std + _mean).permute(1, 2, 0).clamp(0, 1).numpy()


def predict(net, tensor: torch.Tensor, threshold: float):
    with torch.no_grad():
        probs = torch.sigmoid(net(tensor)).squeeze().numpy()
    return probs, (probs > threshold).astype(np.uint8)


def metrics(pred: np.ndarray, gt: np.ndarray):
    eps = 1e-8
    TP  = float(((pred == 1) & (gt == 1)).sum())
    FP  = float(((pred == 1) & (gt == 0)).sum())
    FN  = float(((pred == 0) & (gt == 1)).sum())
    TN  = float(((pred == 0) & (gt == 0)).sum())
    return {
        "Dice":      2 * TP / (2 * TP + FP + FN + eps),
        "IoU":       TP / (TP + FP + FN + eps),
        "Precision": TP / (TP + FP + eps),
        "Recall":    TP / (TP + FN + eps),
        "Accuracy":  (TP + TN) / (TP + TN + FP + FN + eps),
    }


# ──────────────────────────────────────────────────────────────
# Page layout
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠", layout="wide")
st.title("🧠 Brain Tumor Segmentation")
st.write("Upload an MRI scan and the U-Net model will highlight tumor regions.")

# ──────────────────────────────────────────────────────────────
# Sidebar — model info + settings
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model")

    # Download weights from Google Drive if missing.
    # drive.usercontent.google.com with confirm=t bypasses the large-file
    # virus-scan page that urllib and plain requests.get would hit.
    if not os.path.exists(CKPT_PATH):
        gdrive_id = "1lNU1SafiT8nmEDJbMKiqrn1pLLwpw36W"
        url = (
            f"https://drive.usercontent.google.com/download"
            f"?id={gdrive_id}&export=download&confirm=t"
        )
        with st.spinner("Downloading model weights (~120 MB)..."):
            try:
                with requests.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(CKPT_PATH, "wb") as f:
                        for chunk in r.iter_content(chunk_size=32768):
                            f.write(chunk)
                st.cache_resource.clear()
            except Exception as e:
                st.warning(f"Auto-download failed: {e}")

    model, default_threshold, saved_metrics = load_model()

    if model is None:
        st.error("Model file not found. Place `unet_brain_tumor_best.pth` next to app.py.")
    else:
        st.success("Model loaded")
        if saved_metrics:
            st.markdown("**Test-set metrics**")
            c1, c2 = st.columns(2)
            c1.metric("Dice",      f"{saved_metrics.get('f1', 0):.3f}")
            c2.metric("IoU",       f"{saved_metrics.get('iou', 0):.3f}")
            c1.metric("Precision", f"{saved_metrics.get('precision', 0):.3f}")
            c2.metric("Recall",    f"{saved_metrics.get('recall', 0):.3f}")

    st.divider()
    st.header("Settings")
    threshold = st.slider(
        "Detection threshold",
        min_value=0.05, max_value=0.95,
        value=default_threshold, step=0.05,
        help="Lower = catches more tumor (more false positives). Higher = fewer false positives (may miss small tumors).",
    )
    show_heatmap = st.checkbox("Show probability heatmap", value=True)

# ──────────────────────────────────────────────────────────────
# Upload section — always visible
# ──────────────────────────────────────────────────────────────
st.subheader("Upload Image")
col1, col2 = st.columns(2)
with col1:
    mri_file = st.file_uploader(
        "MRI scan (required)",
        type=["tif", "tiff", "jpg", "jpeg", "png"],
        help="Upload a brain MRI image",
    )
with col2:
    gt_file = st.file_uploader(
        "Ground-truth mask (optional)",
        type=["tif", "tiff", "jpg", "jpeg", "png"],
        help="If you have the correct mask, upload it to see Dice / IoU metrics",
    )

# ──────────────────────────────────────────────────────────────
# Inference — only runs when image is uploaded AND model is ready
# ──────────────────────────────────────────────────────────────
if mri_file is None:
    st.info("Upload an MRI scan above to get started.")

elif model is None:
    st.error("Model is not loaded — cannot run inference.")

else:
    img        = Image.open(mri_file)
    img_tensor = preprocess(img)
    probs, pred_mask = predict(model, img_tensor, threshold)
    img_disp   = denormalize(img_tensor)

    tumor_px  = int(pred_mask.sum())
    tumor_pct = 100.0 * tumor_px / (IMG_SIZE * IMG_SIZE)

    st.divider()

    # Banner
    if tumor_px > 0:
        st.error(f"**Tumor detected** — {tumor_pct:.1f}% of scan area ({tumor_px:,} px)")
    else:
        st.success("**No tumor detected** at this threshold.")

    # Ground-truth mask
    gt_mask = None
    if gt_file is not None:
        gt_pil  = Image.open(gt_file).convert("L").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST
        )
        gt_mask = (np.array(gt_pil) > 127).astype(np.uint8)

    # Results grid
    n_cols = 2 + int(show_heatmap) + int(gt_mask is not None)
    cols   = st.columns(n_cols)
    i = 0

    with cols[i]:
        st.subheader("MRI Scan")
        st.image(img_disp, use_container_width=True)
    i += 1

    with cols[i]:
        st.subheader("Predicted Mask")
        st.image(pred_mask.astype(np.float32), use_container_width=True, clamp=True)
    i += 1

    if show_heatmap:
        with cols[i]:
            st.subheader("Probability Map")
            fig, ax = plt.subplots(figsize=(3, 3))
            im = ax.imshow(probs, cmap="hot", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        i += 1

    if gt_mask is not None:
        with cols[i]:
            st.subheader("Overlay vs. GT")
            overlay = img_disp.copy()
            overlay[(pred_mask == 1) & (gt_mask == 1)] = [0.0, 1.0, 0.0]  # TP green
            overlay[(pred_mask == 1) & (gt_mask == 0)] = [1.0, 0.0, 0.0]  # FP red
            overlay[(pred_mask == 0) & (gt_mask == 1)] = [0.0, 0.0, 1.0]  # FN blue
            st.image(overlay, use_container_width=True)
            st.caption("Green = TP  |  Red = FP  |  Blue = FN")
        i += 1

    # Metrics vs GT
    if gt_mask is not None:
        st.divider()
        st.subheader("Metrics vs. Ground Truth")
        m     = metrics(pred_mask, gt_mask)
        mcols = st.columns(len(m))
        for col, (name, val) in zip(mcols, m.items()):
            col.metric(name, f"{val:.4f}")
