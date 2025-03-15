import argparse
import os
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import random

def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")
    return df

def augment_texts(texts, labels, model_name, naug=1, device="cpu"):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    augmented_texts = []
    augmented_labels = []

    for text, label in zip(texts, labels):
        words = text.split()
        if len(words) < 3:
            continue

        for _ in range(naug):
            max_valid_length = min(len(words), 510)
            masked_index = random.randint(0, max_valid_length - 1)
            masked_text = words[:masked_index] + [tokenizer.mask_token] + words[masked_index + 1:]

            masked_sentence = " ".join(masked_text)
            inputs = tokenizer(masked_sentence, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits

            mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
            if mask_token_index.numel() == 0:
                continue

            predicted_token_id = logits[0, mask_token_index].argmax(axis=-1).item()
            predicted_word = tokenizer.decode([predicted_token_id])

            words[masked_index] = predicted_word
            augmented_texts.append(" ".join(words))
            augmented_labels.append(label)

    return augmented_texts, augmented_labels
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='CSV file')
    parser.add_argument('-naug', type=int, default=8, help='Number of augmentation')
    parser.add_argument('--model', type=str, default='bert-base-uncased', help='Contextual Augmentation with BERT')
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    texts, labels = df['text'].tolist(), df['label'].tolist()

    aug_texts, aug_labels = augment_texts(texts, labels, args.model, args.naug)

    df_aug = pd.DataFrame({'text': aug_texts, 'label': aug_labels})
    df_combined = pd.concat([df, df_aug], ignore_index=True)

    base = os.path.splitext(os.path.basename(args.dataset))[0]
    output_path = f"{base}_ca.csv"

    df_combined.to_csv(output_path, index=False)
    print(f"Augmented dataset saved to {output_path}")

if __name__ == '__main__':
    main()
