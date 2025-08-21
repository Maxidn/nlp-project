# Tweet Sentiment Classification with RoBERTa and ELECTRA

This project fine-tunes two transformer-based models, RoBERTa and ELECTRA, on a COVID-19 tweet sentiment classification task. It also applies compression techniques (quantization, pruning, FP16) to reduce model size and improve inference efficiency.

## Task:
Multi-class classification of tweet sentiments into five categories:
- Extremely Negative
- Negative
- Neutral
- Positive
- Extremely Positive

Dataset: [Corona NLP COVID-19 Tweets Dataset (Kaggle)](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification)

## Project Structure

- notebooks/
  
    DL_project_part1.ipynb - EDA & Data Cleaning

    roberta_full_code.ipynb – Fine-tuning + compression using full PyTorch code

    roberta_hf_trainer.ipynb – Fine-tuning + compression using HuggingFace Trainer API

    electra_full_code.ipynb – (to be added)

    electra_hf_trainer.ipynb – (to be added)

  in order to run the notebook corrctly please change the paths for your right locations

- data/

    Corona_NLP_train_clean.csv – Cleaned training set

    Corona_NLP_test_clean.csv – Cleaned test set

- checkpoints/

    roberta_full/

    fp32/ – Full-precision model

    fp16/ – Mixed precision (for GPU inference)

    pruned/ – Unstructured pruning

    quantized/ – Quantized (for CPU inference)

- paper/

    final_paper.pdf

- README.md

- requirements.txt


## Models and Techniques:

- **RoBERTa (cardiffnlp/twitter-roberta-base)** fine-tuned using:
  - Full PyTorch training loop (manual scheduler, early stopping, Optuna tuning)
  - HuggingFace Trainer API

- **Compression methods** applied to best model:
  - **Pruning**: 40% unstructured L1
  - **Quantization**: dynamic quantization with `torch.quantization`
  - **FP16**: model half-precision

## Evaluation:
The model is evaluated using:
- F 1 (macro + weighted) beacuse the data is unbalanced 
- Precision / Recall / Accuracy
- Per-class performance reports

## How to Run:
### COVID-19 Tweet Sentiment — Part 1
A reproducible notebook for exploring and preparing the COVID-19 tweets dataset and cleaning.

1. open "DL_project_part1.ipynb" in jupyter notebook.
2. make sure you have the original Kaggle data (CSV files) and "locations_with_country.xlsx" we add in the data directory
3. IMPORTANT: update the paths below to match where you saved the data on your computer:
<img width="449" height="67" alt="image" src="https://github.com/user-attachments/assets/029a6327-8c75-483d-af59-7632905fc1b9" />
<img width="591" height="49" alt="image" src="https://github.com/user-attachments/assets/9a41c060-0350-4859-9190-26c44d04b2a5" />


### COVID-19 Tweet Sentiment — Part 2
> There are 4 notebooks for each model
> **Recommenץded**: Open notebooks in Google Colab

1. Replace the W&B API key (`wandb.login(...)`) with your own
   
3. 
4. Mount your Google Drive (`drive.mount(...)`) where:
   - Datasets are located (`Corona_NLP_train_clean.csv`, `Corona_NLP_test_clean.csv`)
   - Trained models and compression results are saved
5. files location please change the paths for your right locations


3. Run the cells as-is

## 💾 Files Needed in Google Drive

- `Corona_NLP_train_clean.csv`, `Corona_NLP_test_clean.csv`
- Folder with best fine-tuned RoBERTa and Electra models
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
- Chen Halfi


