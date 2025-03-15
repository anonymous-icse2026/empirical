# An Empirical Study on Data Augmentation for Sentiment Analysis Software: Guidelines for Practitioners

An implementation of the paper "An Empirical Study on Data Augmentation for Sentiment Analysis Software: Guidelines for Practitioners" submitted to the ICSE 2026.

We release our artifacts of Data Augmentation (Genenration) and Fine-tuning to facilitate further resarch and adoption.

## Requirements
```pip install pandas==1.5.3, py-rouge==1.1, scikit-learn==1.2.1, torch==2.1.1, transformers==4.17.0.dev0, tqdm==4.64.1, nltk==3.7, numpy==1.23.5, zss==1.2.0```

-----------
For the first time you run,
```
import nltk

nltk.download('omw-1.4')

nltk.download('wordnet')
```
## Resources
[pretrained model (VGVAE+LC+WN+WPL)](https://drive.google.com/drive/folders/13pii_XG-szMG2KNSuyDn7iPFDyhnXjXm) from the STG's researchers'.

**You have to download `model.ckpt` and `vocab.pkl` in stg folder.**

## Data Augmentation Techniques
We implemented each data augmentation approaches according to their open source codes in GitHub repositories.

These are studies we have studied.
- EDA (Easy Data Augmentation) - [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://aclanthology.org/D19-1670/) 
- AEDA (An Easier Data Augmentation) - [AEDA: An Easier Data Augmentation Technique for Text Classification](https://aclanthology.org/2021.findings-emnlp.234/)
- CA (Contextual Augmentation) - [Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations](https://aclanthology.org/N18-2072/)
- GECA (Good-Enough Compositional Augmentation) - [Good-Enough Compositional Data Augmentation](https://aclanthology.org/2020.acl-main.676/)
- STG (Syntactic Template Generation) - [Controllable Paraphrase Generation with a Syntactic Exemplar](https://arxiv.org/abs/1906.00565)
- TextSmoothing - [Text Smoothing: Enhance Various Data Augmentation Methods on Text Classification Tasks](https://aclanthology.org/2022.acl-short.97/)

```linux
python augment.py -da stg --dataset imdb.csv -naug 8
augment.py has three arguments: -da (eda, aeda, ca, geca, stg and ts),
                                --dataset (cr.csv, subj.csv, sst2.csv, sst5.csv, imdb.csv, trec.csv, etc. from dataset folder),
                                -naug (1, 2, 4, 8, 16, 32, etc.)
```
`augment.py` outputs csv file of augmented datasets in the current directory named `{dataset}_{da}.csv`.


## Evaluation
For the fine-tuning process, we have `train.py`.
```linux
python train.py -m deberta -d imdb_stg.csv
train.py has two arguments: -m (bert, roberta, deberta and distilbert),
                            -d (cr_eda.csv, sst2_geca.csv, imdb_stg.csv, etc.)
```
`train.py` outputs accuracy, precision, recall, and f1-score including classification reports.