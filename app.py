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
GDRIVE_ID = "1YFjIlc4k_pdejMjlz6U_xaVQ6xl59whV"

# ──────────────────────────────────────────────────────────────
# U-Net
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
        self.dec3       = Up(768,  256)
        self.dec2       = Up(384,  128)
        self.dec1       = Up(192,   64)
        self.out_conv   = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        s1=self.enc1(x); s2=self.enc2(s1); s3=self.enc3(s2); s4=self.enc4(s3)
        b=self.bottleneck(s4)
        x=self.dec4(b,s4); x=self.dec3(x,s3); x=self.dec2(x,s2); x=self.dec1(x,s1)
        return self.out_conv(x)

# ──────────────────────────────────────────────────────────────
# Загрузка модели
# ──────────────────────────────────────────────────────────────
def _is_valid_pth(path):
    try:
        if os.path.getsize(path) < 1_000_000: return False
        with open(path, "rb") as f: return f.read(1) == b"\x80"
    except Exception: return False

def _download_weights():
    for url in [
        f"https://drive.usercontent.google.com/download?id={GDRIVE_ID}&export=download&confirm=t",
        f"https://drive.google.com/uc?export=download&id={GDRIVE_ID}&confirm=t",
    ]:
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

# ──────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────
_tf   = T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)), T.ToTensor(),
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
# Конфигурация страницы
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Детектор опухолей мозга",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Базовые стили — убрать лишнее из Streamlit, задать шрифт и цвет фона
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px;}
[data-testid="stTabs"] button {font-size: 0.95rem; font-weight: 500;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Загрузка модели (один раз)
# ──────────────────────────────────────────────────────────────
needs_download = not os.path.exists(CKPT_PATH) or not _is_valid_pth(CKPT_PATH)
model, default_thr, saved_metrics = load_model()

# ──────────────────────────────────────────────────────────────
# Вкладки
# ──────────────────────────────────────────────────────────────
tab_home, tab_demo, tab_about = st.tabs(["Главная", "Демо", "О проекте"])


# ══════════════════════════════════════════════════════════════
# ГЛАВНАЯ
# ══════════════════════════════════════════════════════════════
with tab_home:

    # ── Шапка ────────────────────────────────────────────────
    st.markdown("""
    <div style="padding: 52px 0 40px; border-bottom: 1px solid #21262d; margin-bottom: 2.5rem;">
        <p style="font-size:0.78rem; color:#7d8590; letter-spacing:0.12em;
                  text-transform:uppercase; margin:0 0 1.2rem;">
            Компьютерное зрение &nbsp;/&nbsp; Медицинская визуализация &nbsp;/&nbsp; Сегментация
        </p>
        <h1 style="font-size:clamp(2rem,5vw,3rem); font-weight:700; color:#e6edf3;
                   margin:0 0 1.2rem; line-height:1.2; letter-spacing:-0.02em;">
            Сегментация опухолей<br>головного мозга на МРТ
        </h1>
        <p style="font-size:1.05rem; color:#7d8590; max-width:600px;
                  line-height:1.8; margin:0;">
            U-Net, обученный с нуля на снимках 110 пациентов.
            Модель определяет опухолевые зоны на уровне пикселей —
            без предобученных весов, только архитектура и данные.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Метрики ───────────────────────────────────────────────
    dice = saved_metrics.get("f1",        0.741)
    iou  = saved_metrics.get("iou",       0.589)
    prec = saved_metrics.get("precision", 0.770)
    rec  = saved_metrics.get("recall",    0.715)

    st.markdown("""
    <p style="font-size:0.78rem; color:#7d8590; letter-spacing:0.1em;
              text-transform:uppercase; margin:0 0 1rem;">Результаты на тестовой выборке</p>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val, sub in [
        (c1, "Dice",       f"{dice:.3f}", "основная метрика"),
        (c2, "IoU",        f"{iou:.3f}",  "пересечение / объединение"),
        (c3, "Precision",  f"{prec:.3f}", "точность"),
        (c4, "Recall",     f"{rec:.3f}",  "полнота"),
        (c5, "Параметров", "31.4M",       "в модели"),
    ]:
        col.markdown(f"""
        <div style="border:1px solid #21262d; border-radius:10px;
                    padding:20px 14px; text-align:center; background:#0d1117;">
            <div style="font-size:2rem; font-weight:700; color:#4f8ef7; line-height:1;">{val}</div>
            <div style="font-size:0.82rem; color:#e6edf3; margin-top:6px; font-weight:500;">{label}</div>
            <div style="font-size:0.75rem; color:#484f58; margin-top:3px;">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Как работает ──────────────────────────────────────────
    st.markdown("""
    <p style="font-size:0.78rem; color:#7d8590; letter-spacing:0.1em;
              text-transform:uppercase; margin:2rem 0 1rem;">Как работает</p>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    for col, num, title, text in [
        (s1, "01", "Загрузите снимок",
         "Любой МРТ-снимок в форматах TIFF, JPG или PNG. "
         "Размер и пропорции подгоняются автоматически."),
        (s2, "02", "Нейросеть анализирует",
         "U-Net проходит по кодировщику (encoder), сжимая пространственные признаки, "
         "затем восстанавливает разрешение через декодировщик со skip-связями."),
        (s3, "03", "Получите результат",
         "Бинарная маска опухоли, тепловая карта вероятностей и цветовой overlay "
         "с разметкой TP / FP / FN по пикселям."),
    ]:
        col.markdown(f"""
        <div style="border:1px solid #21262d; border-radius:10px;
                    padding:26px 20px; background:#0d1117; height:100%;">
            <div style="font-size:1.6rem; font-weight:800; color:#4f8ef7; line-height:1;">{num}</div>
            <div style="font-size:0.95rem; font-weight:600; color:#e6edf3;
                        margin:10px 0 8px;">{title}</div>
            <div style="font-size:0.875rem; color:#7d8590; line-height:1.7;">{text}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Датасет ───────────────────────────────────────────────
    st.markdown("""
    <div style="border:1px solid #21262d; border-radius:10px; padding:20px 28px;
                background:#0d1117; margin-top:0.5rem;">
        <p style="color:#7d8590; font-size:0.875rem; margin:0; line-height:1.8;">
            Датасет:
            <strong style="color:#e6edf3;">LGG MRI Segmentation</strong>
            (mateuszbuda / Kaggle)
            &nbsp;&mdash;&nbsp; 7 858 снимков &nbsp;&mdash;&nbsp; 110 пациентов
            &nbsp;&mdash;&nbsp; когорта TCGA, глиомы низкой степени злокачественности
            &nbsp;&mdash;&nbsp; маски размечены вручную экспертами
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ДЕМО
# ══════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown("""
    <h2 style="font-size:1.5rem; font-weight:700; color:#e6edf3; margin:0 0 0.3rem;">
        Загрузите МРТ-снимок</h2>
    <p style="color:#7d8590; font-size:0.9rem; margin:0 0 1.5rem;">
        Модель выделит опухолевую зону на уровне пикселей</p>
    """, unsafe_allow_html=True)

    # ── Скачивание / загрузка модели ─────────────────────────
    if needs_download:
        if os.path.exists(CKPT_PATH):
            os.remove(CKPT_PATH)
        with st.spinner("Загрузка весов модели (~120 МБ)..."):
            ok = _download_weights()
        if ok:
            st.cache_resource.clear()
            st.rerun()
        else:
            st.warning("Автоматическая загрузка не удалась. Загрузите файл вручную.")
            pth_file = st.file_uploader("Загрузить unet_brain_tumor_best.pth", type=["pth"])
            if pth_file:
                with open(CKPT_PATH, "wb") as f:
                    f.write(pth_file.read())
                st.cache_resource.clear()
                st.rerun()
            st.stop()

    if model is None:
        st.error("Модель не загружена. Попробуйте обновить страницу.")
        st.stop()

    # ── Настройки ─────────────────────────────────────────────
    sc1, sc2, _ = st.columns([2, 2, 4])
    with sc1:
        threshold = st.slider("Порог детекции", 0.05, 0.95, float(default_thr), 0.05,
                              help="Ниже — чувствительнее. Выше — точнее.")
    with sc2:
        show_heatmap = st.checkbox("Карта вероятностей", value=True)

    st.divider()

    # ── Загрузчики ────────────────────────────────────────────
    u1, u2 = st.columns(2)
    with u1:
        mri_file = st.file_uploader("МРТ-снимок (обязательно)",
                                    type=["tif","tiff","jpg","jpeg","png"])
    with u2:
        gt_file  = st.file_uploader("Маска разметки (опционально — включает метрики)",
                                    type=["tif","tiff","jpg","jpeg","png"])

    if mri_file is None:
        st.markdown("""
        <div style="border:2px dashed #21262d; border-radius:10px;
                    padding:32px; text-align:center; color:#484f58; font-size:0.9rem;">
            Загрузите снимок выше, чтобы увидеть результат
        </div>""", unsafe_allow_html=True)
        st.stop()

    # ── Инференс ─────────────────────────────────────────────
    img        = Image.open(mri_file)
    img_tensor = preprocess(img)
    probs, pred_mask = infer(model, img_tensor, threshold)
    img_disp   = denormalize(img_tensor)

    tumor_px  = int(pred_mask.sum())
    tumor_pct = 100.0 * tumor_px / (IMG_SIZE * IMG_SIZE)

    if tumor_px > 0:
        st.markdown(
            f'<div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.35);'
            f'border-radius:8px; padding:14px 20px; color:#fca5a5; font-weight:600; font-size:0.95rem;">'
            f'Опухоль обнаружена &mdash; {tumor_pct:.1f}% площади снимка ({tumor_px:,} пикселей)'
            f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background:rgba(34,197,94,0.07); border:1px solid rgba(34,197,94,0.3);'
            'border-radius:8px; padding:14px 20px; color:#86efac; font-weight:600; font-size:0.95rem;">'
            'Опухоль не обнаружена при текущем пороге'
            '</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # GT-маска
    gt_mask = None
    if gt_file:
        gt_pil  = Image.open(gt_file).convert("L").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)
        gt_mask = (np.array(gt_pil) > 127).astype(np.float32)

    # Сетка результатов
    n_cols = 2 + int(show_heatmap) + int(gt_mask is not None)
    cols   = st.columns(n_cols)
    i = 0

    with cols[i]:
        st.caption("МРТ-снимок")
        st.image(img_disp, use_container_width=True)
    i += 1

    with cols[i]:
        st.caption("Предсказанная маска")
        st.image(pred_mask, use_container_width=True, clamp=True)
    i += 1

    if show_heatmap:
        with cols[i]:
            st.caption("Карта вероятностей")
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(probs, cmap="hot", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off"); plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        i += 1

    if gt_mask is not None:
        with cols[i]:
            st.caption("Overlay vs. разметка")
            overlay = img_disp.copy()
            overlay[(pred_mask==1)&(gt_mask==1)] = [0.0, 1.0, 0.0]
            overlay[(pred_mask==1)&(gt_mask==0)] = [1.0, 0.0, 0.0]
            overlay[(pred_mask==0)&(gt_mask==1)] = [0.0, 0.0, 1.0]
            st.image(overlay, use_container_width=True)
            st.caption("Зелёный = TP   |   Красный = FP   |   Синий = FN")
        i += 1

    if gt_mask is not None:
        st.divider()
        st.markdown("**Метрики относительно разметки**")
        m     = calc_metrics(pred_mask, gt_mask)
        mcols = st.columns(len(m))
        for col, (name, val) in zip(mcols, m.items()):
            col.metric(name, f"{val:.4f}")


# ══════════════════════════════════════════════════════════════
# О ПРОЕКТЕ
# ══════════════════════════════════════════════════════════════
with tab_about:

    def card(title, body_html):
        st.markdown(f"""
        <div style="border:1px solid #21262d; border-radius:10px; padding:24px 28px;
                    background:#0d1117; margin-bottom:1rem;">
            <p style="font-size:0.78rem; color:#4f8ef7; letter-spacing:0.1em;
                      text-transform:uppercase; margin:0 0 12px; font-weight:600;">{title}</p>
            {body_html}
        </div>""", unsafe_allow_html=True)

    P = "font-size:0.9rem; color:#c9d1d9; line-height:1.8; margin:0"
    LI = "font-size:0.875rem; color:#c9d1d9; line-height:2;"

    st.markdown("""
    <h2 style="font-size:1.5rem; font-weight:700; color:#e6edf3; margin:0 0 0.3rem;">О проекте</h2>
    <p style="color:#7d8590; font-size:0.9rem; margin:0 0 2rem;">
        Архитектура, датасет, методология обучения</p>
    """, unsafe_allow_html=True)

    # Описание
    card("Постановка задачи", f"""
    <p style="{P}">
        Задача — <strong style="color:#e6edf3;">бинарная сегментация пикселей</strong>:
        каждый пиксель МРТ-снимка классифицируется как <em>опухоль</em> или <em>фон</em>.
        Используется U-Net, обученный с нуля без предобученных весов
        на датасете МРТ 110 пациентов с глиомами низкой степени злокачественности (LGG).
    </p>
    <p style="{P}; margin-top:10px;">
        Ключевое отличие от многих реализаций — разбивка данных <strong style="color:#e6edf3;">по пациентам</strong>,
        а не по снимкам. Это предотвращает утечку данных: снимки одного пациента не попадают
        одновременно в train и test.
    </p>""")

    # Архитектура
    card("Архитектура U-Net", f"""
    <table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
        <tr>
            <th style="background:#161b22; color:#7d8590; padding:9px 14px;
                       text-align:left; font-weight:500; border-bottom:1px solid #21262d;">Блок</th>
            <th style="background:#161b22; color:#7d8590; padding:9px 14px;
                       text-align:left; font-weight:500; border-bottom:1px solid #21262d;">Выходной тензор</th>
            <th style="background:#161b22; color:#7d8590; padding:9px 14px;
                       text-align:left; font-weight:500; border-bottom:1px solid #21262d;">Описание</th>
        </tr>
        <tr style="border-bottom:1px solid #161b22;">
            <td style="padding:9px 14px; color:#c9d1d9;">Вход</td>
            <td style="padding:9px 14px; color:#c9d1d9; font-family:monospace;">[B, 3, 256, 256]</td>
            <td style="padding:9px 14px; color:#7d8590;">RGB МРТ-снимок</td>
        </tr>
        <tr style="background:#0a0e14; border-bottom:1px solid #161b22;">
            <td style="padding:9px 14px; color:#c9d1d9;">Encoder 1</td>
            <td style="padding:9px 14px; color:#c9d1d9; font-family:monospace;">[B, 64, 256, 256]</td>
            <td style="padding:9px 14px; color:#7d8590;">DoubleConv</td>
        </tr>
        <tr style="border-bottom:1px solid #161b22;">
            <td style="padding:9px 14px; color:#c9d1d9;">Encoder 2</td>
            <td style="padding:9px 14px; color:#c9d1d9; font-family:monospace;">[B, 128, 128, 128]</td>
            <td style="padding:9px 14px; color:#7d8590;">MaxPool + DoubleConv</td>
        </tr>
        <tr style="background:#0a0e14; border-bottom:1px solid #161b22;">
            <td style="padding:9px 14px; color:#c9d1d9;">Encoder 3</td>
            <td style="padding:9px 14px; color:#c9d1d9; font-family:monospace;">[B, 256, 64, 64]</td>
            <td style="padding:9px 14px; color:#7d8590;">MaxPool + DoubleConv</td>
        </tr>
        <tr style="border-bottom:1px solid #161b22;">
            <td style="padding:9px 14px; color:#c9d1d9;">Encoder 4</td>
            <td style="padding:9px 14px; color:#c9d1d9; font-family:monospace;">[B, 512, 32, 32]</td>
            <td style="padding:9px 14px; color:#7d8590;">MaxPool + DoubleConv</td>
        </tr>
        <tr style="background:#0a0e14; border-bottom:1px solid #21262d;">
            <td style="padding:9px 14px; color:#4f8ef7; font-weight:600;">Bottleneck</td>
            <td style="padding:9px 14px; color:#4f8ef7; font-family:monospace; font-weight:600;">[B, 1024, 16, 16]</td>
            <td style="padding:9px 14px; color:#4f8ef7;">MaxPool + DoubleConv + Dropout(0.2)</td>
        </tr>
        <tr style="border-bottom:1px solid #161b22;">
            <td style="padding:9px 14px; color:#c9d1d9;">Decoder 4</td>
            <td style="padding:9px 14px; color:#c9d1d9; font-family:monospace;">[B, 512, 32, 32]</td>
            <td style="padding:9px 14px; color:#7d8590;">Upsample + skip + DoubleConv</td>
        </tr>
        <tr style="background:#0a0e14; border-bottom:1px solid #161b22;">
            <td style="padding:9px 14px; color:#c9d1d9;">Decoder 3</td>
            <td style="padding:9px 14px; color:#c9d1d9; font-family:monospace;">[B, 256, 64, 64]</td>
            <td style="padding:9px 14px; color:#7d8590;">Upsample + skip + DoubleConv</td>
        </tr>
        <tr style="border-bottom:1px solid #161b22;">
            <td style="padding:9px 14px; color:#c9d1d9;">Decoder 2</td>
            <td style="padding:9px 14px; color:#c9d1d9; font-family:monospace;">[B, 128, 128, 128]</td>
            <td style="padding:9px 14px; color:#7d8590;">Upsample + skip + DoubleConv</td>
        </tr>
        <tr style="background:#0a0e14; border-bottom:1px solid #161b22;">
            <td style="padding:9px 14px; color:#c9d1d9;">Decoder 1</td>
            <td style="padding:9px 14px; color:#c9d1d9; font-family:monospace;">[B, 64, 256, 256]</td>
            <td style="padding:9px 14px; color:#7d8590;">Upsample + skip + DoubleConv</td>
        </tr>
        <tr>
            <td style="padding:9px 14px; color:#4f8ef7; font-weight:600;">Выход</td>
            <td style="padding:9px 14px; color:#4f8ef7; font-family:monospace; font-weight:600;">[B, 1, 256, 256]</td>
            <td style="padding:9px 14px; color:#4f8ef7;">Conv 1&times;1 — логиты (без сигмоиды)</td>
        </tr>
    </table>
    <p style="{P}; margin-top:14px; color:#7d8590; font-size:0.82rem;">
        Всего параметров: 31 396 627 &nbsp;&mdash;&nbsp;
        Dropout только в боттлнеке для регуляризации &nbsp;&mdash;&nbsp;
        Skip-связи передают детали с энкодера в декодер
    </p>""")

    # Два блока рядом
    col_a, col_b = st.columns(2)

    with col_a:
        card("Датасет", f"""
        <ul style="padding-left:1.1rem; margin:0;">
            <li style="{LI}"><strong style="color:#e6edf3;">Источник:</strong> Kaggle — mateuszbuda/lgg-mri-segmentation</li>
            <li style="{LI}"><strong style="color:#e6edf3;">Когорта:</strong> TCGA, глиомы низкой степени (LGG)</li>
            <li style="{LI}"><strong style="color:#e6edf3;">Снимков:</strong> 7 858</li>
            <li style="{LI}"><strong style="color:#e6edf3;">Пациентов:</strong> 110 уникальных</li>
            <li style="{LI}"><strong style="color:#e6edf3;">Маски:</strong> ручная разметка экспертами</li>
            <li style="{LI}"><strong style="color:#e6edf3;">Формат:</strong> TIFF, 256&times;256 px</li>
            <li style="{LI}"><strong style="color:#e6edf3;">Дисбаланс:</strong> ~80% снимков без опухоли</li>
        </ul>""")

        card("Аугментация (только train)", f"""
        <ul style="padding-left:1.1rem; margin:0;">
            <li style="{LI}">Горизонтальное / вертикальное отражение</li>
            <li style="{LI}">Поворот &plusmn;15&deg;</li>
            <li style="{LI}">Случайный кроп + ресайз (зум-ин)</li>
            <li style="{LI}">Яркость / контраст (только снимок, не маска)</li>
            <li style="{LI}">Все трансформации применяются к снимку и маске с одними параметрами</li>
        </ul>""")

    with col_b:
        card("Параметры обучения", f"""
        <ul style="padding-left:1.1rem; margin:0;">
            <li style="{LI}"><strong style="color:#e6edf3;">Разбивка:</strong> 70% train / 15% val / 15% test <em>по пациентам</em></li>
            <li style="{LI}"><strong style="color:#e6edf3;">Оптимизатор:</strong> Adam, lr=1e-4, weight_decay=1e-5</li>
            <li style="{LI}"><strong style="color:#e6edf3;">Scheduler:</strong> ReduceLROnPlateau (&times;0.5 при плато)</li>
            <li style="{LI}"><strong style="color:#e6edf3;">Функция потерь:</strong> BCEWithLogitsLoss + Dice (50/50)</li>
            <li style="{LI}"><strong style="color:#e6edf3;">Батч:</strong> 16</li>
            <li style="{LI}"><strong style="color:#e6edf3;">Эпохи:</strong> до 50, EarlyStopping (patience=7)</li>
            <li style="{LI}"><strong style="color:#e6edf3;">Порог:</strong> подбор на val-выборке (sweep 0.05–0.95)</li>
            <li style="{LI}"><strong style="color:#e6edf3;">Среда:</strong> Google Colab, GPU T4</li>
        </ul>""")

        card("Функция потерь", f"""
        <p style="{P}">
            <strong style="color:#e6edf3;">BCEWithLogitsLoss</strong> — численно стабильная версия BCE,
            сигмоида применяется внутри. Корректно работает с несбалансированными классами.<br><br>
            <strong style="color:#e6edf3;">Dice Loss</strong> — напрямую оптимизирует метрику перекрытия,
            помогает при дисбалансе классов (много пикселей фона).<br><br>
            Итоговая: <code style="background:#161b22; padding:2px 6px; border-radius:4px;
            color:#79c0ff;">0.5 &times; BCE + 0.5 &times; Dice</code>
        </p>""")

    # Технологии
    card("Стек технологий", "".join([
        f'<span style="display:inline-block; background:#161b22; border:1px solid #30363d;'
        f'color:#79c0ff; padding:4px 14px; border-radius:20px; font-size:0.82rem;'
        f'margin:3px 4px 3px 0;">{t}</span>'
        for t in ["PyTorch", "torchvision", "Streamlit", "scikit-learn",
                  "NumPy", "Pillow", "Matplotlib", "Google Colab", "Python 3.10"]
    ]))
