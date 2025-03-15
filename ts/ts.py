import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
import re


def clean_decoded_text(text):
    text = re.sub(r"\[unused\d+\]", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ##", "")
    return text


def augment_sentence(sentence, model, tokenizer, embedding_layer, lambda_val=0.7, device='cpu'):

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    vocab_size = tokenizer.vocab_size

    one_hot = F.one_hot(input_ids, num_classes=vocab_size).float()

    with torch.no_grad():
        hidden_states = model.bert(**inputs).last_hidden_state

    logits = torch.matmul(hidden_states, embedding_layer.weight.t())
    mlm_prob = torch.softmax(logits, dim=-1)

    smoothed = lambda_val * one_hot + (1 - lambda_val) * mlm_prob

    for idx in range(smoothed.size(1)):
        token_prob = smoothed[0, idx]
        for unused_token in range(999, 1050):
            token_prob[unused_token] = 0

    augmented_ids = []
    for i in range(smoothed.size(1)):
        orig_token_id = input_ids[0, i].item()
        if orig_token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            augmented_ids.append(orig_token_id)
        else:
            token_prob = smoothed[0, i]
            sampled_id = torch.multinomial(token_prob, num_samples=1).item()
            augmented_ids.append(sampled_id)

    augmented_sentence = tokenizer.decode(augmented_ids, skip_special_tokens=True)
    return clean_decoded_text(augmented_sentence)


def main():
    parser = argparse.ArgumentParser(description="Text Smoothing")
    parser.add_argument("--dataset", type=str, required=True, help="CSV file")
    parser.add_argument("-naug", type=int, default=8, help="number of augmentation")
    args = parser.parse_args()

    dataset_path = os.path.join("dataset", args.dataset) if os.path.basename(
        args.dataset) == args.dataset else args.dataset

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV needs 'text', 'label'")

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    embedding_layer = model.bert.embeddings.word_embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    augmented_data = []
    for idx, row in df.iterrows():
        orig_text = row["text"]
        label = row["label"]
        augmented_data.append({"text": orig_text, "label": label})
        for _ in range(args.naug):
            aug_text = augment_sentence(orig_text, model, tokenizer, embedding_layer, lambda_val=0.7, device=device)
            augmented_data.append({"text": aug_text, "label": label})

    base = os.path.splitext(os.path.basename(dataset_path))[0]
    output_path = f"{base}_ts.csv"
    aug_df = pd.DataFrame(augmented_data)
    aug_df.to_csv(output_path, index=False)
    print(f"Augmented dataset saved as {output_path}")


if __name__ == "__main__":
    main()
