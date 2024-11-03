# 🤖Transformer-en-fr-Translator
# Transformer-based English-to-French Translation Model
<h2> EN -> FR</h2>
This repository contains the code and resources for a transformer-based model designed to translate English text into French. The model was trained on parallel English and French datasets, leveraging a transformer architecture to achieve high-quality translation results.

## 🔎Project Overview

The primary objective of this project is to build and fine-tune a transformer-based neural network for machine translation. The model leverages attention mechanisms to improve translation accuracy and handles complex language structures effectively. It was trained over multiple epochs, reaching a significant level of accuracy at epoch 85.

### 🛠️Model Architecture

The model is based on a transformer architecture, which is particularly well-suited for tasks involving sequence-to-sequence data, such as machine translation. Key components include:

- **Attention Mechanism:** The model uses multi-head self-attention layers to capture dependencies in both the source (English) and target (French) sequences.
- **Positional Encoding:** Positional information is added to the input embeddings to capture the order of tokens in each sequence.
- **Layer Stacking:** Multiple encoder and decoder layers enable the model to learn complex linguistic patterns.

## 🗂️Dataset

The model was trained on a parallel English-French dataset, processed into tokens and further encoded to create input-output pairs for supervised learning.

### 🔠Vocabulary

The English and French datasets were tokenized and encoded separately, yielding vocabularies of approximately:

- **English Vocabulary:** ~15,000 words (based on `spaCy` tokenization)
- **French Vocabulary:** ~26,000 words (based on `spaCy` tokenization)

## 📝Training Details

The model was trained for **85 epochs**. The training process involved optimizing cross-entropy loss between the predicted and actual token sequences. Training was conducted on a Kaggle GPU instance with **CUDA 10.2** support.

## 🚩Model Checkpoint

The trained model for epoch 85 is available for download at the following link:

[85th Epoch Translation Model](https://www.kaggle.com/datasets/swayamshah09/85-th-epoch-translation-model-saved)

To load the model, download the checkpoint file and follow the steps in the provided Jupyter notebook.

## 📄Repository Contents

- **`transformer-en-fr-translator.ipynb`**: The Jupyter notebook containing the complete model code, from preprocessing to model training and evaluation.
- **`README.md`**: Overview and description of the project.

## 📈Usage

1. Clone the repository and navigate to the project directory.
3. Download the model checkpoint from the link above.
4. Open and run the Jupyter notebook `transformer-en-fr-translator.ipynb` to explore the model training and evaluation steps.

## 🎯Results and Evaluation

The model achieved notable accuracy in translating English to French sentences, showing robust performance across a range of sentence structures and vocabulary. Evaluation metrics and sample translations are included in the notebook.

## 🤝Acknowledgments

Special thanks to the developers of `torch` and `transformers` for providing the essential tools for building this model.
