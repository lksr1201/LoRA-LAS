

---

# Open-Vocabulary Speech Emotion Recognition with Label-Aware Scaling LoRA (LoRA-LAS)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

This repository provides the official implementation of **LoRA-LAS** (Label-Aware Scaling Low-Rank Adaptation), an improved parameter-efficient fine-tuning method for **Open-Vocabulary Speech Emotion Recognition (OV-SER)**, as described in the paper:

> **"基于改进低阶适应的开放词汇语音情感识别"**  
> Liu Kangwei, Ding Hanyu, Gao Lijian, Mao Qirong  
> *Journal of Frontiers of Computer Science and Technology*

[[Paper Link (arXiv/TBD)](https://doi.org/10.3778/j.issn.1673-9418.0000000)]  

---

## 📌 Overview

Traditional speech emotion recognition (SER) relies on closed-set categorical labels (e.g., "happy", "angry"). In contrast, **open-vocabulary SER** leverages **Audio Large Language Models (AudioLLMs)** and **Large Language Models (LLMs)** to generate rich, natural-language emotion descriptions (e.g., "frustrated yet hopeful", "nervous excitement").

However, standard LoRA fine-tuning under low-rank constraints struggles to capture the **nonlinear, long-tailed nature** of open-vocabulary emotion semantics. To address this, we propose **LoRA-LAS** — a novel LoRA variant that injects **emotion label semantics** into a **nonlinear scaling function**, enabling dynamic, label-aware adaptation with minimal extra parameters.

---

## ✨ Key Features

- ✅ **Label-aware nonlinear LoRA**: Introduces a **LAS module** that scales low-rank updates using emotion context embeddings.
- ✅ **End-to-end OV-SER pipeline**: Combines AudioLLM (e.g., SALMONN) for **acoustic description generation** + LLM (e.g., Qwen2.5-7B) for **emotion captioning**.
- ✅ **Parameter-efficient**: Only ~1% of LLM parameters are trainable (e.g., 10M params at rank=4).
- ✅ **Strong generalization**: Works on both **open-vocabulary** (MER-Caption Plus, OV-MERD) and **traditional categorical** datasets (IEMOCAP, MELD, CMU-MOSI).
- ✅ **Outperforms SOTA**: Beats AffectGPT, LoRA+, DoRA, and PiSSA with fewer parameters.

---

## 📂 Datasets

| Dataset           | Type               | #Samples | #Emotions | Annotation Style              |
|-------------------|--------------------|----------|-----------|-------------------------------|
| MER-Caption Plus  | Open-vocabulary    | 31,327   | 1,972     | Model-led + Human-assisted   |
| OV-MERD           | Open-vocabulary    | 332      | 236       | Human-led + Model-assisted   |
| IEMOCAP           | Categorical        | 6,433    | 6         | Human                         |
| MELD              | Categorical        | 13,708   | 7         | Human                         |
| CMU-MOSI          | Multi-label        | 2,197    | Continuous| Regression / Multi-label      |

> 🔸 We train on **MER-Caption Plus** and evaluate on **OV-MERD**, **IEMOCAP**, **MELD**, and **CMU-MOSI**.

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/your-username/LoRA-LAS.git
cd LoRA-LAS

# Create virtual environment (optional but recommended)
conda create -n lora-las python=3.9
conda activate lora-las

# Install dependencies
pip install -r requirements.txt
```

### Requirements (`requirements.txt`)

```txt
torch>=2.0
transformers>=4.35
accelerate>=0.25
datasets>=2.14
numpy
scikit-learn
tqdm
```

> 💡 AudioLLM (SALMONN) and LLM (Qwen2.5-7B) weights are loaded via Hugging Face Hub.

---

## ▶️ Quick Start

### 1. Generate Audio Descriptions (Offline)



### 2. Train LoRA-LAS



### 3. Evaluate


---

## 📊 Results (WF / HiT)

| Method                     | Rank | OV-MERD (WF) | MELD (HiT) | IEMOCAP (HiT) | CMU-MOSI (WF) |
|----------------------------|------|--------------|------------|---------------|----------------|
| AffectGPT (LoRA)           | 8    | 59.55        | 56.27      | 60.14         | 81.11          |
| + DoRA                     | 8    | 60.70        | 58.44      | 60.88         | 81.30          |
| **+ LoRA-LAS (Ours)**      | **8**| **61.07**    | **60.76**  | **61.21**     | **81.97**      |
| **+ LoRA-LAS (Ours)**      | **4**| **60.73**    | —          | —             | —              |

> ✅ **LoRA-LAS at rank=4** outperforms **AffectGPT at rank=8** with **50% fewer trainable parameters**.

---

## 📐 Evaluation Metrics

- **Open-vocabulary**:  
  - **Precision**, **Recall**, **Weighted F1 (WF)**  
  - Labels normalized via **3-stage grouping** (lemmatization → synonym merging → emotion wheel mapping)
- **Categorical**:  
  - **Hit Rate (HiT)**: 1 if any predicted label (after normalization) matches ground truth

---

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{liu2025lora-las,
  title={基于改进低阶适应的开放词汇语音情感识别},
  author={Liu, Kangwei and Ding, Hanyu and Gao, Lijian and Mao, Qirong},
  
}
```

---

## 🙏 Acknowledgements

- Supported by National Natural Science Foundation of China (Grants No. 62176106, 62576155)
- Built upon [LoRA](https://arxiv.org/abs/2106.09685), [AffectGPT](https://arxiv.org/abs/2501.16566), [SALMONN](https://arxiv.org/abs/2308.16838), and [Qwen2.5](https://arxiv.org/abs/2412.15115)

---

## 📧 Contact

For questions or collaboration, please contact the corresponding author:  
**Prof. Qirong Mao** — `Mao_qr@ujs.edu.cn`

---

> 🚀 We welcome contributions, issues, and PRs!
