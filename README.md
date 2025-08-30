# Multimodal-Emotion-Recognition-with-MELD

This project implements a deep learning model for emotion classification using the **MELD (Multimodal EmotionLines Dataset)**. It fuses **textual**, **audio**, and **visual** features to predict emotions in conversations using a custom PyTorch pipeline.

## 🚀 Features

- BERT-based text embeddings
- GRU-based temporal modeling for audio and video
- Attention pooling for modality-specific feature aggregation
- Fusion architecture for emotion classification
- Training loop with checkpointing and evaluation metrics

## 📦 Dataset

- [MELD.Raw.tar.gz](http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz)
- Preprocessed features loaded from `train.pkl`

## 🧠 Model Architecture

- Text: BERT + Linear projection
- Audio/Video: Bidirectional GRU + Attention pooling
- Fusion: Concatenation + Fully connected layers

## 📊 Evaluation

- Accuracy and weighted F1-score
- Classification report with per-class metrics

## 🛠️ Installation

```bash
pip install -r requirements.txt
---

## 📦 requirements.txt

```txt
torch
transformers
scikit-learn
pandas
numpy
tqdm
gdown

