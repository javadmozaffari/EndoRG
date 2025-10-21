# EndoRG: Automated Endoscopic Report Generation with Multimodal AI through Dataset, Framework and Benchmarking
### This repository contains the code and dataset for the paper *"EndoRG: Automated Endoscopic Report Generation with Multimodal AI through Dataset, Framework and Benchmarking".*

---

## 🧠 Overview
EndoRG introduces a multimodal framework for **automated clinical report generation** in gastrointestinal (GI) endoscopy.  
The project provides:
- A **new endoscopic dataset** with 7,438 image–report pairs across ten anatomical regions.
- A **vision–language model** combining:
  - Visual Mamba (Vmamba) backbone  
  - Knowledge Distillation from a ViT-Small teacher trained on **GastroNet-5M**  
  - Contrastive Learning (CL) for cross-modal alignment  
  - Anatomical Classification (CLS) module for location grounding  
  - Context Sample Retrieval (CSR) for case-based reasoning  
  - **TinyLlama** as the lightweight report-generation backbone  

The framework achieves superior **semantic fidelity (CIDEr)** while maintaining strong linguistic fluency.

---

## 🖼️ Dataset and Qualitative Examples

https://drive.google.com/file/d/1G-QBCalHEkfPpgY42zYnkg1iVRfn41Lz/view?usp=sharing

<p align="center">
  <img src="docs/figures/fig3_dataset_overview.png" alt="Overview of the EndoRG dataset showing anatomical diversity across GI regions" width="85%">
</p>
<p align="center"><b>Figure 3.</b> Anatomical diversity in the EndoRG dataset. Representative endoscopic images and excerpts from corresponding clinical reports across the GI tract.</p>

---

<p align="center">
  <img src="docs/figures/fig4_qualitative_comparison.png" alt="Qualitative comparison of generated endoscopic reports" width="85%">
</p>
<p align="center"><b>Figure 4.</b> Qualitative comparison of generated reports. EndoRG produces clinically faithful and anatomically consistent outputs compared to prior models.</p>

---

## 📈 Performance Comparison of LLM Backbones (Table 4)

| Model              | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | METEOR | CIDEr  |
|--------------------|:------:|:------:|:------:|:------:|:--------:|:-------:|:------:|
| TinyLlama          | **0.4321** | **0.3386** | **0.2811** | **0.2382** | **0.4532** | **0.2107** | **1.2384** |
| Qwen1.5-0.5B       | 0.4157 | 0.3256 | 0.2694 | 0.2287 | 0.4549 | 0.2057 | 1.1636 |
| Qwen1.5-1.8B       | 0.4412 | 0.3392 | 0.2768 | 0.2315 | 0.4364 | 0.2082 | 1.0650 |
| Phi-1.5            | 0.4026 | 0.3255 | 0.2720 | 0.2320 | 0.4239 | 0.2043 | 1.1484 |
| LLM2CLIP-1B        | 0.3804 | 0.3051 | 0.2539 | 0.2157 | 0.4076 | 0.1983 | 1.1281 |

<p align="center"><b>Table 4.</b> Performance comparison of lightweight LLMs for endoscopic report generation.</p>

---


## 📦 Dataset Access and Structure

The **EndoRG Dataset** consists of **7,438 endoscopic images** and paired clinical reports covering ten anatomical regions of the gastrointestinal (GI) tract:

> **Regions:**  
> *esophagus, fundus & cardia, body, antrum, duodenum, ascending colon, transverse colon, descending colon, sigmoid, rectum*

The dataset is organized into **training** and **test** subsets to support supervised learning and benchmarking.

### 📁 Folder Structure

    EndoRGData
    |-- Train
    |   |-- esophagus
    |   |-- fundus_and_cardia
    |   |-- body
    |   |-- antrum
    |   |-- duodenum
    |   |-- ascending_colon
    |   |-- transverse_colon
    |   |-- descending_colon
    |   |-- sigmoid
    |   |-- rectum
    |   |-- train_reports.xlsx
    |-- Test
    |   |-- esophagus
    |   |-- fundus_and_cardia
    |   |-- body
    |   |-- antrum
    |   |-- duodenum
    |   |-- ascending_colon
    |   |-- transverse_colon
    |   |-- descending_colon
    |   |-- sigmoid
    |   |-- rectum
    |   |-- test_reports.xlsx


    
Each folder contains endoscopic images specific to that anatomical region.  
The corresponding clinical descriptions are stored in the Excel files under the `reports/` directory, where each row links the **image filename** to its **structured report text**.

---

### 🔗 Dataset Download
The dataset is available for research use under the **CC BY 4.0 license**.

- 📥 **[Download from Zenodo](https://zenodo.org/your-dataset-link)**  
  *(or replace with your actual Google Drive / Hugging Face / institutional link)*

If you use the dataset in your research, please cite our paper using the BibTeX entry below.

---

### ⚠️ Data Usage Terms
- The dataset is intended **solely for non-commercial research and educational purposes**.  
- Redistribution without proper attribution is prohibited.  
- Please cite:  
  *“EndoRG: Automated Endoscopic Report Generation with Multimodal AI through Dataset, Framework and Benchmarking.”*

---
