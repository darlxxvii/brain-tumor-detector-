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
# Config
# ──────────────────────────────────────────────────────────────
IMG_SIZE  = 256
CKPT_PATH = "unet_brain_tumor_best.pth"
GDRIVE_ID = "1lNU1SafiT8nmEDJbMKiqrn1pLLwpw36W"

# ──────────────────────────────────────────────────────────────
# U-Net (must match train.py)
# ──────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                   nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, dropout))
    def forward(self, x): return self.block(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        dh, dw = skip.size(2)-x.size(2), skip.size(3)-x.size(3)
        x = F.pad(x, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        return self.conv(torch.cat([skip, x], dim=1))

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1       = DoubleConv(3, 64)
        self.enc2       = Down(64, 128)
        self.enc3       = Down(128, 256)
        self.enc4       = Down(256, 512)
        self.bottleneck = Down(512, 1024, dropout=0.2)
        self.dec4       = Up(1536, 512)
        self.dec3       = Up(768, 256)
        self.dec2       = Up(384, 128)
        self.dec1       = Up(192, 64)
        self.out_conv   = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        s1=self.enc1(x); s2=self.enc2(s1); s3=self.enc3(s2); s4=self.enc4(s3)
        b=self.bottleneck(s4)
        x=self.dec4(b,s4); x=self.dec3(x,s3); x=self.dec2(x,s2); x=self.dec1(x,s1)
        return self.out_conv(x)

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _is_valid_pth(path):
    try:
        if os.path.getsize(path) < 1_000_000: return False
        with open(path, "rb") as f: return f.read(1) == b"\x80"
    except Exception: return False

def _download_weights():
    urls = [
        f"https://drive.usercontent.google.com/download?id={GDRIVE_ID}&export=download&confirm=t",
        f"https://drive.google.com/uc?export=download&id={GDRIVE_ID}&confirm=t",
    ]
    for url in urls:
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(CKPT_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536): f.write(chunk)
            if _is_valid_pth(CKPT_PATH): return True
        except Exception: pass
    return False

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(CKPT_PATH) or not _is_valid_pth(CKPT_PATH):
        return None, 0.5, {}
    try:
        ckpt  = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        net   = UNet()
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
        net.load_state_dict(state)
        net.eval()
        thr = float(ckpt.get("threshold", 0.5)) if isinstance(ckpt, dict) else 0.5
        m   = ckpt.get("test_metrics", {})       if isinstance(ckpt, dict) else {}
        return net, thr, m
    except Exception: return None, 0.5, {}

_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                 T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
_mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
_std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

def preprocess(img): return _tf(img.convert("RGB")).unsqueeze(0)
def denormalize(t):  return (t.squeeze()*_std+_mean).permute(1,2,0).clamp(0,1).numpy()

def infer(net, tensor, thr):
    with torch.no_grad():
        probs = torch.sigmoid(net(tensor)).squeeze().numpy()
    return probs, (probs > thr).astype(np.float32)

def calc_metrics(pred, gt):
    eps=1e-8
    TP=float(((pred==1)&(gt==1)).sum()); FP=float(((pred==1)&(gt==0)).sum())
    FN=float(((pred==0)&(gt==1)).sum()); TN=float(((pred==0)&(gt==0)).sum())
    return {"Dice":2*TP/(2*TP+FP+FN+eps), "IoU":TP/(TP+FP+FN+eps),
            "Precision":TP/(TP+FP+eps),   "Recall":TP/(TP+FN+eps),
            "Accuracy":(TP+TN)/(TP+TN+FP+FN+eps)}

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BrainScan AI — Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #0c2340 60%, #0e3460 100%);
    border-radius: 20px; padding: 64px 48px; text-align: center;
    margin-bottom: 2rem; position: relative; overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(56,189,248,0.08) 0%, transparent 70%);
}
.hero-tag {
    display: inline-block; background: rgba(56,189,248,0.15);
    color: #38bdf8; border: 1px solid rgba(56,189,248,0.3);
    padding: 5px 16px; border-radius: 20px; font-size: 0.82rem;
    letter-spacing: .06em; text-transform: uppercase; margin-bottom: 1.2rem;
}
.hero h1 {
    font-size: clamp(2rem, 5vw, 3.4rem); font-weight: 800;
    color: #f1f5f9; margin: 0 0 1rem; line-height: 1.15;
}
.hero h1 span { color: #38bdf8; }
.hero p {
    font-size: 1.1rem; color: #94a3b8; max-width: 580px;
    margin: 0 auto 2rem; line-height: 1.7;
}

/* ── Stat cards ── */
.stat-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 14px; padding: 24px 16px; text-align: center;
}
.stat-value { font-size: 2.2rem; font-weight: 700; color: #38bdf8; }
.stat-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }

/* ── Section title ── */
.section-title {
    font-size: 1.6rem; font-weight: 700; color: #f1f5f9;
    margin: 2rem 0 0.3rem;
}
.section-sub { font-size: 0.95rem; color: #64748b; margin-bottom: 1.5rem; }

/* ── Step cards ── */
.step-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 14px; padding: 28px 24px; height: 100%;
}
.step-num { font-size: 2rem; font-weight: 800; color: #38bdf8; line-height:1; }
.step-title { font-size: 1rem; font-weight: 600; color: #f1f5f9; margin: 10px 0 6px; }
.step-text  { font-size: 0.88rem; color: #94a3b8; line-height: 1.6; }

/* ── Info cards (About) ── */
.info-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 14px; padding: 24px 28px; margin-bottom: 1rem;
}
.info-card h3 { font-size: 1rem; font-weight: 600; color: #38bdf8; margin: 0 0 10px; }
.info-card p, .info-card li {
    font-size: 0.9rem; color: #cbd5e1; line-height: 1.7; margin: 0;
}
.info-card ul { padding-left: 1.2rem; margin: 0; }

/* ── Architecture table ── */
.arch { width:100%; border-collapse:collapse; font-size:0.85rem; }
.arch th { background:#334155; color:#94a3b8; padding:10px 14px; text-align:left; font-weight:500; }
.arch td { padding:9px 14px; border-bottom:1px solid #273447; color:#cbd5e1; }
.arch tr:nth-child(even) td { background:#1a2538; }
.arch .highlight td { color:#38bdf8; font-weight:600; }

/* ── Demo upload area ── */
.upload-hint {
    background: #1e293b; border: 2px dashed #334155;
    border-radius: 12px; padding: 20px; text-align: center;
    color: #64748b; font-size: 0.9rem;
}

/* ── Result banner ── */
.tumor-banner {
    background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.4);
    border-radius: 10px; padding: 14px 20px; color: #fca5a5;
    font-size: 1rem; font-weight: 600;
}
.clear-banner {
    background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3);
    border-radius: 10px; padding: 14px 20px; color: #86efac;
    font-size: 1rem; font-weight: 600;
}

/* ── Tech badges ── */
.badge {
    display: inline-block; background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.25); color: #7dd3fc;
    padding: 4px 12px; border-radius: 20px; font-size: 0.8rem;
    margin: 3px 3px 3px 0;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Model loading (runs once, cached)
# ──────────────────────────────────────────────────────────────
needs_download = not os.path.exists(CKPT_PATH) or not _is_valid_pth(CKPT_PATH)
model, default_thr, saved_metrics = load_model()

# ──────────────────────────────────────────────────────────────
# Navigation tabs
# ──────────────────────────────────────────────────────────────
tab_home, tab_demo, tab_about = st.tabs(["🏠  Home", "🔬  Demo", "📖  About"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════════════════════
with tab_home:

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-tag">Deep Learning · Medical Imaging · Segmentation</div>
        <h1>AI-Powered <span>Brain Tumor</span><br>Detection</h1>
        <p>
            Upload a brain MRI scan and our U-Net model will precisely
            segment tumor regions at the pixel level — in under a second.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Performance metrics
    dice  = saved_metrics.get("f1",         0.741)
    iou   = saved_metrics.get("iou",        0.589)
    prec  = saved_metrics.get("precision",  0.770)
    rec   = saved_metrics.get("recall",     0.715)

    st.markdown('<p class="section-title">Model Performance</p>'
                '<p class="section-sub">Evaluated on held-out test patients (patient-level split, no leakage)</p>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val in [
        (c1, "Dice Score",  f"{dice:.3f}"),
        (c2, "IoU",         f"{iou:.3f}"),
        (c3, "Precision",   f"{prec:.3f}"),
        (c4, "Recall",      f"{rec:.3f}"),
        (c5, "Parameters",  "31.4M"),
    ]:
        col.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{val}</div>
            <div class="stat-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown('<p class="section-title">How It Works</p>'
                '<p class="section-sub">Three steps from upload to diagnosis</p>',
                unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    for col, num, title, text in [
        (s1, "01", "Upload MRI Scan",
         "Upload any brain MRI image in TIFF, JPG, or PNG format. "
         "The model accepts standard clinical scan formats."),
        (s2, "02", "AI Segmentation",
         "Our U-Net architecture analyzes every pixel of the 256×256 scan "
         "through encoder-decoder paths with skip connections."),
        (s3, "03", "Instant Results",
         "Get a binary segmentation mask, probability heatmap, and overlay "
         "showing tumor regions highlighted in real time."),
    ]:
        col.markdown(f"""
        <div class="step-card">
            <div class="step-num">{num}</div>
            <div class="step-title">{title}</div>
            <div class="step-text">{text}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset banner
    st.markdown("""
    <div class="info-card" style="text-align:center; border-color:rgba(56,189,248,0.2);">
        <p style="color:#94a3b8; margin:0; font-size:0.9rem;">
            Trained on the
            <strong style="color:#38bdf8;">LGG MRI Segmentation dataset</strong>
            &nbsp;·&nbsp; 7,858 brain MRI scans
            &nbsp;·&nbsp; 110 unique patients
            &nbsp;·&nbsp; TCGA lower-grade glioma cohort
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — DEMO
# ══════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown('<p class="section-title">Live Demo</p>'
                '<p class="section-sub">Upload a brain MRI scan to detect and segment tumor regions</p>',
                unsafe_allow_html=True)

    # ── Model download / status ──────────────────────────────
    if needs_download:
        if os.path.exists(CKPT_PATH):
            os.remove(CKPT_PATH)
        with st.spinner("Downloading model weights (~120 MB) — first load only..."):
            ok = _download_weights()
        if ok:
            st.cache_resource.clear()
            st.rerun()
        else:
            st.warning(
                "Auto-download failed. "
                "Please upload the model file manually."
            )
            uploaded_pth = st.file_uploader(
                "Upload unet_brain_tumor_best.pth",
                type=["pth"], key="pth_upload"
            )
            if uploaded_pth:
                with open(CKPT_PATH, "wb") as f:
                    f.write(uploaded_pth.read())
                st.cache_resource.clear()
                st.rerun()
            st.stop()

    if model is None:
        st.error("Model failed to load. Try refreshing the page.")
        st.stop()

    # ── Settings ─────────────────────────────────────────────
    col_set1, col_set2, _ = st.columns([2, 2, 4])
    with col_set1:
        threshold = st.slider(
            "Detection threshold", 0.05, 0.95,
            float(default_thr), 0.05,
            help="Lower = more sensitive  |  Higher = more precise",
        )
    with col_set2:
        show_heatmap = st.checkbox("Show probability heatmap", value=True)

    st.divider()

    # ── File uploaders ────────────────────────────────────────
    up1, up2 = st.columns(2)
    with up1:
        mri_file = st.file_uploader(
            "MRI Scan (required)",
            type=["tif", "tiff", "jpg", "jpeg", "png"],
        )
    with up2:
        gt_file = st.file_uploader(
            "Ground-truth mask (optional — enables metrics)",
            type=["tif", "tiff", "jpg", "jpeg", "png"],
        )

    if mri_file is None:
        st.markdown("""
        <div class="upload-hint">
            Upload an MRI scan above to see the segmentation result
        </div>""", unsafe_allow_html=True)
        st.stop()

    # ── Inference ────────────────────────────────────────────
    img        = Image.open(mri_file)
    img_tensor = preprocess(img)
    probs, pred_mask = infer(model, img_tensor, threshold)
    img_disp   = denormalize(img_tensor)

    tumor_px  = int(pred_mask.sum())
    tumor_pct = 100.0 * tumor_px / (IMG_SIZE * IMG_SIZE)

    if tumor_px > 0:
        st.markdown(
            f'<div class="tumor-banner">Tumor detected &nbsp;—&nbsp; '
            f'{tumor_pct:.1f}% of scan area &nbsp;({tumor_px:,} px)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="clear-banner">No tumor detected at the current threshold</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Ground-truth
    gt_mask = None
    if gt_file:
        gt_pil  = Image.open(gt_file).convert("L").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)
        gt_mask = (np.array(gt_pil) > 127).astype(np.float32)

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
        st.image(pred_mask, use_container_width=True, clamp=True)
    i += 1

    if show_heatmap:
        with cols[i]:
            st.subheader("Probability Map")
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(probs, cmap="hot", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off"); plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        i += 1

    if gt_mask is not None:
        with cols[i]:
            st.subheader("Overlay vs. GT")
            overlay = img_disp.copy()
            overlay[(pred_mask==1)&(gt_mask==1)] = [0.0, 1.0, 0.0]
            overlay[(pred_mask==1)&(gt_mask==0)] = [1.0, 0.0, 0.0]
            overlay[(pred_mask==0)&(gt_mask==1)] = [0.0, 0.0, 1.0]
            st.image(overlay, use_container_width=True)
            st.caption("Green = TP  |  Red = FP  |  Blue = FN")
        i += 1

    # Metrics vs GT
    if gt_mask is not None:
        st.divider()
        st.subheader("Metrics vs. Ground Truth")
        m     = calc_metrics(pred_mask, gt_mask)
        mcols = st.columns(len(m))
        for col, (name, val) in zip(mcols, m.items()):
            col.metric(name, f"{val:.4f}")


# ══════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════
with tab_about:
    st.markdown('<p class="section-title">About the Project</p>'
                '<p class="section-sub">Architecture, dataset, and training methodology</p>',
                unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        # Overview
        st.markdown("""
        <div class="info-card">
            <h3>Project Overview</h3>
            <p>
                This project trains a <strong>U-Net segmentation model</strong> from scratch
                to detect and segment low-grade glioma (LGG) brain tumors in MRI scans.
                The model performs binary pixel-level classification: each pixel is labeled
                as <em>tumor</em> or <em>background</em>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Architecture table
        st.markdown("""
        <div class="info-card">
            <h3>U-Net Architecture</h3>
            <table class="arch">
                <tr><th>Layer</th><th>Output shape</th><th>Details</th></tr>
                <tr><td>Input</td><td>[B, 3, 256, 256]</td><td>RGB brain MRI</td></tr>
                <tr><td>Encoder 1</td><td>[B, 64, 256, 256]</td><td>DoubleConv</td></tr>
                <tr><td>Encoder 2</td><td>[B, 128, 128, 128]</td><td>MaxPool + DoubleConv</td></tr>
                <tr><td>Encoder 3</td><td>[B, 256, 64, 64]</td><td>MaxPool + DoubleConv</td></tr>
                <tr><td>Encoder 4</td><td>[B, 512, 32, 32]</td><td>MaxPool + DoubleConv</td></tr>
                <tr class="highlight"><td>Bottleneck</td><td>[B, 1024, 16, 16]</td><td>MaxPool + DoubleConv + Dropout(0.2)</td></tr>
                <tr><td>Decoder 4</td><td>[B, 512, 32, 32]</td><td>Upsample + skip + DoubleConv</td></tr>
                <tr><td>Decoder 3</td><td>[B, 256, 64, 64]</td><td>Upsample + skip + DoubleConv</td></tr>
                <tr><td>Decoder 2</td><td>[B, 128, 128, 128]</td><td>Upsample + skip + DoubleConv</td></tr>
                <tr><td>Decoder 1</td><td>[B, 64, 256, 256]</td><td>Upsample + skip + DoubleConv</td></tr>
                <tr class="highlight"><td>Output</td><td>[B, 1, 256, 256]</td><td>Conv 1×1 — logits (no sigmoid)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Loss
        st.markdown("""
        <div class="info-card">
            <h3>Loss Function</h3>
            <p>
                Combined <strong>BCEWithLogitsLoss + Dice Loss</strong> (50/50 weight).<br><br>
                BCEWithLogitsLoss applies sigmoid internally for numerical stability.
                Dice Loss directly optimizes the overlap metric. Together they handle
                class imbalance (most pixels are background) while keeping gradients stable.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        # Dataset
        st.markdown("""
        <div class="info-card">
            <h3>Dataset</h3>
            <ul>
                <li><strong>Source:</strong> mateuszbuda/lgg-mri-segmentation (Kaggle)</li>
                <li><strong>Cohort:</strong> TCGA lower-grade glioma (LGG)</li>
                <li><strong>Images:</strong> 7,858 brain MRI slices</li>
                <li><strong>Patients:</strong> 110 unique patients</li>
                <li><strong>Masks:</strong> expert-annotated binary</li>
                <li><strong>Format:</strong> TIFF, 256×256</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Training
        st.markdown("""
        <div class="info-card">
            <h3>Training Setup</h3>
            <ul>
                <li><strong>Split:</strong> 70% train / 15% val / 15% test<br>
                    <em>by patient</em> (no cross-patient leakage)</li>
                <li><strong>Optimizer:</strong> Adam (lr=1e-4, weight_decay=1e-5)</li>
                <li><strong>Scheduler:</strong> ReduceLROnPlateau (×0.5 on plateau)</li>
                <li><strong>Epochs:</strong> up to 50 with EarlyStopping (patience=7)</li>
                <li><strong>Batch size:</strong> 16</li>
                <li><strong>Threshold:</strong> tuned on val set (sweep 0.1–0.9)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Augmentation
        st.markdown("""
        <div class="info-card">
            <h3>Augmentation (train only)</h3>
            <ul>
                <li>Horizontal &amp; vertical flip</li>
                <li>Rotation ±15°</li>
                <li>Random zoom-in (crop + resize)</li>
                <li>Brightness / contrast jitter</li>
                <li><em>All transforms applied jointly to image and mask</em></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Tech stack
        st.markdown("""
        <div class="info-card">
            <h3>Technologies</h3>
            <span class="badge">PyTorch</span>
            <span class="badge">torchvision</span>
            <span class="badge">Streamlit</span>
            <span class="badge">scikit-learn</span>
            <span class="badge">NumPy</span>
            <span class="badge">Pillow</span>
            <span class="badge">Matplotlib</span>
            <span class="badge">Google Colab</span>
        </div>
        """, unsafe_allow_html=True)
