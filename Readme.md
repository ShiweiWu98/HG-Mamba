<h1 align="center">HG-Mamba</h1>

<p align="center">
  <strong>Heuristic-Guided State Space Model for Laparoscopic Image Desmoking</strong>
</p>

<p align="center">
  Shiwei Wu, Xiaobo Zhu, Song Zhang, Yu An, Xinyu Zhang, Hui Yan, Jie Tian, Xiyuan Hu, Zhenyu Liu
</p>

<p align="center">
  <em>ICME 2026</em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx">📄 Paper</a> |
  <a href="https://github.com/yourname/HG-Mamba">💻 Code</a> |
  <a href="https://drive.google.com/xxx">🤗 Pretrained Models</a> |
  <a href="https://xxx">📦 Dataset</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red" />
  <img src="https://img.shields.io/badge/ICME-2026-green" />
  <img src="https://img.shields.io/badge/Task-Laparoscopic%20Desmoking-orange" />
</p>

---

## 🔥 Overview

Surgical smoke generated during laparoscopic procedures severely degrades image visibility and affects intraoperative decision-making.  
We propose **HG-Mamba**, a novel and lightweight state space model backbone for **laparoscopic image desmoking**.

HG-Mamba improves conventional Mamba from both the **spatial** and **frequency** perspectives:

- **HG-SSM**: introduces **input-guided dynamic sampling** and **heuristic-guided state fusion** for adaptive spatial context modeling.
- **FR-FFN**: performs **multi-band frequency decomposition** with adaptive weighting to enhance frequency-aware representation.

Together, these designs enable HG-Mamba to effectively remove complex surgical smoke while remaining highly lightweight.

---

## ✨ Highlights

- 🚀 A novel **Heuristic-Guided State Space Model (HG-SSM)** for flexible context modeling beyond sequential state transitions.
- 🎯 A **Frequency Refine Feed-Forward Network (FR-FFN)** for fine-grained frequency modulation.
- ⚡ **Only 1.69M parameters**, reducing parameter count by **82.45%** compared with **MambaIRv2-S (9.63M)**.
- 🏆 Superior performance on both **synthetic** and **real-world** laparoscopic desmoking benchmarks.
- 📦 A large-scale **synthetic smoke dataset** is constructed to supplement limited real paired data.

---

## 🖼️ Visual Results

### Qualitative comparison
<p align="center">
  <img src="assets/visual_results.png" width="90%">
</p>

HG-Mamba produces cleaner reconstructions and preserves sharper anatomical boundaries under diverse smoke conditions.

---

## 🏗️ Framework

<p align="center">
  <img src="assets/framework.png" width="90%">
</p>

The overall architecture of our HG-Mamba-based desmoking network.

---

## 📊 Main Results

### Quantitative comparison on Synthetic Dataset and DesmokeData

## 📊 Quantitative Results

Comparison on the **Synthetic Dataset** and **DesmokeData**.  
Higher **SSIM / PSNR** indicates better performance, while lower **CIEDE / Params / MACs** is preferred.

| Method         | Venue       | Syn. SSIM ↑ | Syn. PSNR ↑ | Syn. CIEDE ↓ | Desmoke SSIM ↑ | Desmoke PSNR ↑ | Desmoke CIEDE ↓ |  Params ↓ |     MACs ↓ |
| -------------- | ----------- | ----------: | ----------: | -----------: | -------------: | -------------: | --------------: | --------: | ---------: |
| Cyclic-DeGAN   | CBM'20      |       0.865 |      21.280 |        7.525 |          0.833 |         22.397 |           7.173 |    11.97M |     28.11G |
| DehazeFormer-B | TIP'23      |       0.959 |      28.219 |        3.159 |          0.879 |         26.341 |           4.511 |     2.52M |     19.76G |
| DEA-Net        | TIP'23      |       0.964 |      29.706 |        2.833 |          0.847 |         25.747 |           4.714 |     3.65M |     24.68G |
| SelfSVD        | ECCV'24     |       0.964 |      29.412 |        2.705 |          0.870 |         25.670 |           4.681 |    15.58M |     23.51G |
| ConvIR-S       | TPAMI'24    |       0.953 |      28.849 |        2.891 |          0.877 |         26.442 |           4.540 |     5.53M |     42.23G |
| MB-Taylor-B V2 | TPAMI'25    |       0.969 |      29.551 |        2.658 |          0.878 |         26.515 |           4.383 |     2.63M |     24.40G |
| SGDN           | AAAI'25     |       0.970 |      29.670 |        2.605 |          0.879 |         26.871 |           4.168 |    11.09M |     41.16G |
| MambaIRv2-S    | CVPR'25     |       0.964 |      28.730 |        2.814 |          0.870 |         25.234 |           4.891 |     9.63M |    192.91G |
| **Ours**       | **ICME'26** |   **0.972** |  **30.393** |    **2.354** |      **0.888** |     **27.072** |       **4.064** | **1.69M** | **18.62G** |

---

## 🧠 Method

### 1. Heuristic-Guided State Space Model (HG-SSM)
HG-SSM introduces an input-guided dynamic sampling strategy to select spatially related hidden states and fuse them adaptively, overcoming the limitations of strict sequential state transitions in conventional SSMs.

### 2. Frequency Refine Feed-Forward Network (FR-FFN)
FR-FFN decomposes features into multiple frequency bands and applies adaptive weighting to each band, enabling finer frequency refinement and better structure restoration.

## 🛠️ Installation

Please follow the steps below to set up the environment.  
The code has been tested with the following configuration:

- Python 3.10.16
- CUDA 11.8
- torch==2.0.1
- torchvision==0.15.2
- torchaudio==2.0.2

### 1. Clone the repository
```bash
git clone https://github.com/yourname/HG-Mamba.git
cd HG-Mamba
```

### 2.Create a conda environment

```bash
conda create -n HG_Mamba python=3.10.16 -y
conda activate HG_Mamba
```

### 3.Install PyTorch with CUDA 11.8

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### 4.Install other dependencies and selective scan kernel

```bash
pip install --upgrade pip
pip install -r requirements.txt
cd kernels/selective_scan
pip install .
```

## 📦 Dataset Preparation

Please organize the dataset in the following structure:

```bash
data/
├── train/
│   ├── smoky/         # smoky images
│   └── smokeless/     # corresponding smoke-free images with the same filenames
├── val/
│   ├── smoky/
│   └── smokeless/
└── test/
    ├── smoky/
    └── smokeless/
```

**Example:**

```
train/
├── smoky/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── smokeless/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

## 🚀 Training

To train the model, run:

```
python common_trainer.py --config_file configs/config.yaml --mode train
```

## 🧪 Testing

To evaluate the model, run:

```
python common_trainer.py --config_file configs/config.yaml --mode test
```

------

## 🖼️ Inference

To run inference and save restored images, use:

```
python common_trainer.py --config_file configs/config.yaml --mode predict
```