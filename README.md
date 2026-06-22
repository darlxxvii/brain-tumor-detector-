# Brain Tumor Segmentation using U-Net

An end-to-end Deep Learning pipeline for binary pixel-level segmentation of brain tumors on Magnetic Resonance Imaging (MRI) scans using a custom U-Net architecture implemented from scratch in PyTorch.

## Key Features & Architecture Highlights

Unlike standard baseline implementations, this repository addresses critical medical imaging and machine learning challenges:

1. **Patient-Level Data Split:** To prevent severe data leakage, the dataset is stratified and split (70% Train, 15% Val, 15% Test) strictly by unique patient IDs, ensuring the model is tested on entirely unseen anatomy.
2. **Advanced Joint Augmentation:** Implements synchronous transformations (rotations, flips, zoom/crop) on both raw images and target masks, utilizing NEAREST interpolation for masks to maintain crisp pixel boundaries.
3. **Robust Loss & Numerical Stability:** Combines BCEWithLogitsLoss and a custom DiceLoss (50/50 ratio). The model outputs raw logits rather than probabilities, improving gradient stability.
4. **Regularization:** Built-in Dropout2d(0.2) in the architecture's bottleneck layer alongside Adam weight_decay to mitigate overfitting in the 31M-parameter model.
5. **Dynamic Validation Controls:** Features dynamic learning rate scaling (ReduceLROnPlateau) and EarlyStopping (patience=10) triggered by validation performance.
6. **Post-Processing Threshold Tuning:** Automatically sweeps thresholds on the validation set to find the optimal decision boundary for maximum Dice Score, avoiding an arbitrary 0.5 cutoff.
7. **Tumor-Only Metrics:** Evaluates final metrics specifically on frames containing tumors to showcase actual model capability, avoiding inflation from easy "all-black" background slices.

---

## Project Structure

```text
├── train.ipynb               # Research, training pipeline, and evaluation notebook
├── train.py                  # Python script equivalent for automated training runs
├── app.py                    # Streamlit web application script for deployment
├── requirements.txt          # Python dependencies
├── unet_brain_tumor_best.pth # Saved model checkpoint (Weights & Metadata)
└── README.md                 # Project documentation

```

---

## Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/darlxxvii/brain-tumor-detector-.git
cd brain-tumor-detector-

```



```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

```

3. **Dataset:**
The script automatically fetches the [mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) dataset via `kagglehub`.

---

## Training the Model

You can run the full training pipeline, check training curves, and execute threshold tuning inside the Jupyter Notebook:

```bash
jupyter notebook train.ipynb

```

Or run it as a standalone script:

```bash
python train.py

```

*The model tracks validation loss and saves the best performing weights into `unet_brain_tumor_best.pth`.*

---

## Web App Deployment (Streamlit)

To visualize the model's predictions via a local web browser GUI, run the pre-configured Streamlit app:

```bash
streamlit run app.py

```

### How it works:

1. Upload any patient's brain MRI scan (`.tif`, `.png`, or `.jpg`).
2. The system loads the saved U-Net weights and optimal threshold metadata.
3. It instantly displays the raw scan, the generated prediction mask, and an **Error Overlay** (Green = True Positive, Red = False Positive, Blue = False Negative).
