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
    """Валидный .pth: ZIP-архив (PK, PyTorch 1.6+) или legacy pickle (0x80).
    HTML-страница ошибки начинается с '<' (0x3C)."""
    try:
        if os.path.getsize(path) < 1_000_000:
            return False
        with open(path, "rb") as f:
            head = f.read(2)
        return head[:2] == b"PK" or head[:1] == b"\x80"
    except Exception:
        return False

def _download_weights(progress_bar=None, log=None):
    """Скачивает веса модели. Возвращает (успех, сообщение об ошибке)."""
    urls = [
        "https://github.com/darlxxvii/brain-tumor-detector-/releases/download/v1.0/unet_brain_tumor_best.pth",
        f"https://drive.usercontent.google.com/download?id={GDRIVE_ID}&export=download&confirm=t",
        f"https://drive.google.com/uc?export=download&id={GDRIVE_ID}&confirm=t",
    ]
    errors = []
    for url in urls:
        source = url.split("/")[2]  # github.com / drive.usercontent...
        if log: log.info(f"Пробуем: {source}...")
        try:
            with requests.get(url, stream=True, timeout=300, allow_redirects=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(CKPT_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_bar and total:
                            progress_bar.progress(
                                min(downloaded / total, 1.0),
                                text=f"{source}: {downloaded//1_048_576} / {total//1_048_576} МБ"
                            )
            size_mb = os.path.getsize(CKPT_PATH) / 1_048_576
            if _is_valid_pth(CKPT_PATH):
                if log: log.success(f"Скачано {size_mb:.1f} МБ с {source}")
                return True, ""
            # невалидный файл (HTML вместо .pth)
            err = f"{source}: скачано {size_mb:.1f} МБ, но файл невалидный (HTML?)"
            errors.append(err)
            if log: log.warning(err)
            os.remove(CKPT_PATH)
        except Exception as e:
            err = f"{source}: {type(e).__name__}: {e}"
            errors.append(err)
            if log: log.error(err)
    return False, " | ".join(errors)

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
    # ВАЖНО: не используем st.stop() — он останавливает весь скрипт,
    # и вкладка "О проекте" (идёт ниже по коду) не отрисовывается.
    # Вместо этого вся логика обёрнута во вложенные if-блоки.
    demo_ready = True

    if needs_download:
        if os.path.exists(CKPT_PATH):
            os.remove(CKPT_PATH)
        st.info("Первый запуск: скачиваем веса модели (~120 МБ)...")
        log_area = st.empty()
        pb = st.progress(0, text="Подключаемся...")

        class _Log:
            def info(self, msg):    log_area.info(msg)
            def success(self, msg): log_area.success(msg)
            def warning(self, msg): log_area.warning(msg)
            def error(self, msg):   log_area.error(msg)

        ok, err_msg = _download_weights(progress_bar=pb, log=_Log())
        pb.empty()

        if ok:
            st.cache_resource.clear()
            st.rerun()
        else:
            log_area.error(f"Все источники недоступны: {err_msg}")
            st.warning("Загрузите файл вручную:")
            pth_file = st.file_uploader("unet_brain_tumor_best.pth", type=["pth"])
            if pth_file:
                with open(CKPT_PATH, "wb") as f:
                    f.write(pth_file.read())
                st.cache_resource.clear()
                st.rerun()
            demo_ready = False

    if demo_ready and model is None:
        st.error("Модель не загружена. Попробуйте обновить страницу.")
        demo_ready = False

    if demo_ready:
        # ── Настройки ─────────────────────────────────────────
        sc1, sc2, _ = st.columns([2, 2, 4])
        with sc1:
            threshold = st.slider("Порог детекции", 0.05, 0.95, float(default_thr), 0.05,
                                  help="Ниже — чувствительнее. Выше — точнее.")
        with sc2:
            show_heatmap = st.checkbox("Карта вероятностей", value=True)

        st.divider()

        # ── Загрузчики ─────────────────────────────────────────
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
        else:
            # ── Инференс ───────────────────────────────────────
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
    import pandas as pd

    st.subheader("О проекте")
    st.caption("Архитектура, датасет, методология обучения")
    st.divider()

    # ── Постановка задачи ────────────────────────────────────
    st.markdown("#### Постановка задачи")
    st.markdown("""
Задача — **бинарная сегментация пикселей**: каждый пиксель МРТ-снимка
классифицируется как *опухоль* или *фон*.
Используется U-Net, обученный с нуля без предобученных весов
на датасете МРТ 110 пациентов с глиомами низкой степени злокачественности (LGG).

Ключевое отличие — разбивка данных **по пациентам**, а не по снимкам.
Это предотвращает утечку данных: снимки одного пациента не попадают
одновременно в train и test.
""")
    st.divider()

    # ── Архитектура ──────────────────────────────────────────
    st.markdown("#### Архитектура U-Net")
    arch_df = pd.DataFrame([
        ["Вход",        "[B, 3, 256, 256]",    "RGB МРТ-снимок"],
        ["Encoder 1",   "[B, 64, 256, 256]",   "DoubleConv"],
        ["Encoder 2",   "[B, 128, 128, 128]",  "MaxPool + DoubleConv"],
        ["Encoder 3",   "[B, 256, 64, 64]",    "MaxPool + DoubleConv"],
        ["Encoder 4",   "[B, 512, 32, 32]",    "MaxPool + DoubleConv"],
        ["Bottleneck",  "[B, 1024, 16, 16]",   "MaxPool + DoubleConv + Dropout(0.2)  ← регуляризация"],
        ["Decoder 4",   "[B, 512, 32, 32]",    "Upsample + skip + DoubleConv"],
        ["Decoder 3",   "[B, 256, 64, 64]",    "Upsample + skip + DoubleConv"],
        ["Decoder 2",   "[B, 128, 128, 128]",  "Upsample + skip + DoubleConv"],
        ["Decoder 1",   "[B, 64, 256, 256]",   "Upsample + skip + DoubleConv"],
        ["Выход",       "[B, 1, 256, 256]",    "Conv 1x1 — логиты (без сигмоиды)"],
    ], columns=["Блок", "Выходной тензор", "Описание"])
    st.dataframe(arch_df, use_container_width=True, hide_index=True)
    st.caption("Всего параметров: 31 396 627  |  Skip-связи передают детали с энкодера в декодер")
    st.divider()

    # ── Две колонки: датасет + обучение ─────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Датасет")
        st.markdown("""
- **Источник:** Kaggle — mateuszbuda/lgg-mri-segmentation
- **Когорта:** TCGA, глиомы низкой степени (LGG)
- **Снимков:** 7 858
- **Пациентов:** 110 уникальных
- **Маски:** ручная разметка экспертами
- **Формат:** TIFF, 256×256 px
- **Дисбаланс:** ~80% снимков без опухоли
""")

        st.markdown("#### Аугментация (только train)")
        st.markdown("""
- Горизонтальное / вертикальное отражение
- Поворот ±15°
- Случайный кроп + ресайз (зум-ин)
- Яркость / контраст — только снимок, не маска
- Все трансформации применяются к снимку и маске с одними параметрами
""")

    with col_b:
        st.markdown("#### Параметры обучения")
        st.markdown("""
- **Разбивка:** 70% train / 15% val / 15% test *по пациентам*
- **Оптимизатор:** Adam, lr=1e-4, weight_decay=1e-5
- **Scheduler:** ReduceLROnPlateau (×0.5 при плато)
- **Функция потерь:** BCEWithLogitsLoss + Dice (50/50)
- **Батч:** 16
- **Эпохи:** до 50, EarlyStopping (patience=7)
- **Порог:** подбор на val-выборке (sweep 0.05–0.95)
- **Среда:** Google Colab, GPU T4
""")

        st.markdown("#### Функция потерь")
        st.markdown("""
**BCEWithLogitsLoss** — численно стабильная версия BCE,
сигмоида применяется внутри. Корректно работает с несбалансированными классами.

**Dice Loss** — напрямую оптимизирует метрику перекрытия,
помогает при дисбалансе классов (много пикселей фона).

Итоговая: `Loss = 0.5 × BCE + 0.5 × Dice`
""")

    st.divider()

    # ── Стек технологий ──────────────────────────────────────
    st.markdown("#### Стек технологий")
    techs = ["PyTorch", "torchvision", "Streamlit", "scikit-learn",
             "NumPy", "Pillow", "Matplotlib", "Google Colab", "Python 3.10"]
    st.markdown("  ".join([f"`{t}`" for t in techs]))
