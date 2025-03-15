import argparse
import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from transformers import DebertaTokenizer, DebertaForSequenceClassification, AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import re
import sys


class CustomDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file, encoding='utf-8')
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.labels = self.data['label'].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        if isinstance(label, str):
            label_idx = torch.tensor(self.labels.tolist().index(label))  # CR dataset
        else:
            label_idx = torch.tensor(label)
        combined_text = f"{text}"

        encoding = self.tokenizer(combined_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return input_ids, attention_mask, label_idx

def evaluate_model(model, dataloader, criterion):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_samples += len(labels)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / num_samples

    class_report = classification_report(all_labels, all_preds, target_names=[str(label) for label in dataset.labels])

    return accuracy, f1, precision, recall, avg_loss, class_report


# Argument parser to take model name as an option
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, help="Choose model: bert, roberta, deberta, distilbert")
parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset file name (e.g., test.csv)")
args = parser.parse_args()

if args.model == 'bert':
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model_class = BertForSequenceClassification
elif args.model == 'roberta':
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model_class = RobertaForSequenceClassification
elif args.model == 'deberta':
    model_name = 'microsoft/deberta-base'
    tokenizer = DebertaTokenizer.from_pretrained(model_name)
    model_class = DebertaForSequenceClassification
elif args.model == 'distilbert':
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model_class = DistilBertForSequenceClassification
else:
    print("Invalid model name. Choose from 'bert', 'roberta', 'deberta', or 'distilbert'.")
    sys.exit()

csv_file_path = args.dataset
#csv_file_path = os.path.join('dataset/', args.dataset)
dataset = CustomDataset(csv_file=csv_file_path, tokenizer=tokenizer)
batch_size = 16
shuffle = True
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
csv_file = re.split(r'[.]', args.dataset)[0]
dataset_name = csv_file.rsplit('_',1)[0]
print(f"Using model: {model_name}")
print(f"Using dataset: {dataset_name}")

if dataset_name == 'trec':
    test_csv_file_path = "dataset/trec_test.csv"
    print(test_csv_file_path)
    test_dataset = CustomDataset(csv_file=test_csv_file_path, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=93)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
elif dataset_name == 'mpqa':
    test_csv_file_path = "dataset/mpqa_test.csv"
    print(test_csv_file_path)
    test_dataset = CustomDataset(csv_file=test_csv_file_path, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=93)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
elif dataset_name == 'sst5':
    test_csv_file_path = "dataset/sst5_test.csv"
    print(test_csv_file_path)
    test_dataset = CustomDataset(csv_file=test_csv_file_path, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=93)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
elif dataset_name == 'sst2':
    test_csv_file_path = "dataset/sst2_test.csv"
    print(test_csv_file_path)
    test_dataset = CustomDataset(csv_file=test_csv_file_path, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=93)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
elif dataset_name == 'imdb':
    test_csv_file_path = "dataset/imdb_test.csv"
    print(test_csv_file_path)
    test_dataset = CustomDataset(csv_file=test_csv_file_path, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=93)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
elif dataset_name == 'imdb_mul':
    test_csv_file_path = "dataset/imdb_mul_test.csv"
    print(test_csv_file_path)
    test_dataset = CustomDataset(csv_file=test_csv_file_path, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=93)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
elif dataset_name == 'cr':
    test_csv_file_path = "dataset/cr_test.csv"
    print(test_csv_file_path)
    test_dataset = CustomDataset(csv_file=test_csv_file_path, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=93)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
else:
    train_data, val_data = train_test_split(dataset, test_size=0.4, random_state=93)
    val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=93)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

if model_name == 'bert-base-uncased':
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(dataset.labels))
elif model_name == 'roberta-base':
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(dataset.labels))
elif model_name == 'microsoft/deberta-base':
    model = DebertaForSequenceClassification.from_pretrained(model_name, num_labels=len(dataset.labels))
elif model_name == 'distilbert-base-uncased':
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(dataset.labels))
else:
    print("model name is invalid")
    sys.exit()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model.to(device)

criterions = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-7, no_deprecation_warning=True)

train_accuracy_history = []
train_f1_history = []
val_accuracy_history = []
val_f1_history = []

# Early stopping parameters
num_epochs = 200
patience, best_f1 = 20, 0.0
early_stop_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for input_ids, attention_mask, labels in train_dataloader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    train_acc, train_f1, _, _, _, _ = evaluate_model(model, train_dataloader, criterions)
    val_acc, val_f1, val_prec, val_rec, val_loss, _ = evaluate_model(model, val_dataloader, criterions)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    print(f"Validation - Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")

    # Early Stopping Logic
    if val_f1 > best_f1:
        best_f1 = val_f1
        early_stop_counter = 0
        # Optionally, save the best model here
    else:
        early_stop_counter += 1
        print(f"No improvement. Early Stop Counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        break

# val_acc, val_f1, val_prec, val_rec, val_loss, val_report = evaluate_model(model, val_dataloader, criterions)
# print(f"Validation - Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
# print("\nValidation Report:\n", val_report)

# Test evaluation
test_acc, test_f1, test_prec, test_rec, test_loss, test_report = evaluate_model(model, test_dataloader, criterions)
print(f"Test Set - Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}")
print("\nTest Report:\n", test_report)

# Save final test results to a file
# with open('output.txt', 'a') as file:
#     file.write(f"{model_name} - {csv_file_path} | Test Results: Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}\n")
#     file.write(f"Classification Report:\n{test_report}\n")
