# Tweet Sentiment Classification with RoBERTa and ELECTRA

This project fine-tunes two transformer-based models — RoBERTa and ELECTRA — on a COVID-19 tweet sentiment classification task. It also applies compression techniques (quantization, pruning, FP16) to reduce model size and improve inference efficiency.

## 🧠 Task
Multi-class classification of tweet sentiments into five categories:
- Extremely Negative
- Negative
- Neutral
- Positive
- Extremely Positive

Dataset: [Corona NLP COVID-19 Tweets Dataset (Kaggle)](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification)

## Project Structure

- notebooks/

    roberta_full_code.ipynb – Fine-tuning + compression using full PyTorch code

    roberta_hf_trainer.ipynb – Fine-tuning + compression using HuggingFace Trainer API

    electra_full_code.ipynb – (to be added)

    electra_hf_trainer.ipynb – (to be added)

- data/

    train.csv – Cleaned training set

    test.csv – Cleaned test set

    checkpoints/

    roberta_full/

    fp32/ – Full-precision model

    fp16/ – Mixed precision (for GPU inference)

    pruned/ – Unstructured pruning

    quantized/ – Quantized (for CPU inference)

- paper/

    final_paper.pdf

- README.md

- requirements.txt (optional)


## 🧪 Models and Techniques

- **RoBERTa (cardiffnlp/twitter-roberta-base)** fine-tuned using:
  - Full PyTorch training loop (manual scheduler, early stopping, Optuna tuning)
  - HuggingFace Trainer API

- **Compression methods** applied to best model:
  - **Pruning**: 40% unstructured L1
  - **Quantization**: dynamic quantization with `torch.quantization`
  - **FP16**: model half-precision

## 📊 Evaluation
The model is evaluated using:
- Accuracy
- Precision / Recall / F1 (macro + weighted)
- Per-class performance reports

## 📍 How to Run

> 💡 **Recommended**: Open notebooks in Google Colab

1. Replace the W&B API key (`wandb.login(...)`) with your own  
2. Mount your Google Drive (`drive.mount(...)`) where:
   - Datasets are located (`Corona_NLP_train_clean.csv`, `Corona_NLP_test_clean.csv`)
   - Trained models and compression results are saved

3. Run the cells as-is

## 💾 Files Needed in Google Drive

- `Corona_NLP_train_clean.csv`, `Corona_NLP_test_clean.csv`
- Folder with best fine-tuned RoBERTa model
- Folder with compressed models:
  - `/fp16/`, `/pruned/`, `/quantized/` with saved models
  - `load_quantized.py` is included to help re-load quantized models

## 📎 Notes

- Outputs are visible in notebooks for review
- `wandb` tracking logs hyperparameters, metrics, and trials
- Optuna is used for tuning learning rate, dropout, batch size, etc.
- The same architecture is applied to ELECTRA (coming soon)

## 👥 Authors
- Maximiliano Niemetz
- Chen Ben Halfi

---
For questions or issues, please open an issue or contact the authors.
