import pandas as pd
import numpy as np
import random
from nltk.corpus import wordnet
import nltk
import argparse
import os

nltk.download('wordnet')

class Vocab:
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = []

    def add(self, word):
        if word not in self.word_to_index:
            self.word_to_index[word] = len(self.index_to_word)
            self.index_to_word.append(word)

    def decode(self, seq):
        return [self.index_to_word[int(idx)] for idx in seq if isinstance(idx, int) and idx < len(self.index_to_word)]

class OneShotDataset:
    def __init__(self, train_utts, val_utts, test_utts, aug_data=(), invert=False):
        vocab = Vocab()
        for i in range(5):
            vocab.add(f"WUG{i}")
        vocab.add("##")
        for utts in (train_utts, val_utts, test_utts):
            for inp, out in utts:
                for word in inp:
                    vocab.add(word)

        self.vocab = vocab
        self.sep = "##"
        self.train_utts = train_utts
        self.aug_utts = aug_data
        self.val_utts = val_utts
        self.test_utts = test_utts

    def get_synonym(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name().lower() != word.lower():
                    synonyms.add(lemma.name().replace('_', ' '))
        return random.choice(list(synonyms)) if synonyms else None

    def enumerate_comp(self, replace_ratio):
        for inp, out in self.train_utts:
            num_replacements = max(1, int(len(inp) * replace_ratio))
            candidate_words = [word for word in inp if self.get_synonym(word)]
            if candidate_words:
                selected_words = random.sample(candidate_words, min(num_replacements, len(candidate_words)))
                names = {}
                for original_word in selected_words:
                    new_word = self.get_synonym(original_word)
                    if new_word:
                        idx = inp.index(original_word)
                        inp[idx] = new_word
                        names[original_word] = new_word
                yield (inp, out), names
            else:
                yield (inp, out), {}

    def realize(self, seq, names):
        return [names.get(word, word) for word in seq], None

class GECAProcessor:
    def __init__(self, input_csv, output_csv, replace_ratio=0.15):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.replace_ratio = replace_ratio
        self.dataset = None

    def load_data(self):
        df = pd.read_csv(self.input_csv, encoding='utf-8')
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV file must contain 'text' and 'label' columns.")
        return df

    def prepare_dataset(self, df):
        train_data = list(zip(df['text'].apply(str.split), df['label'].astype(str)))
        self.dataset = OneShotDataset(train_data, [], [])

    def augment_data(self, num_samples=10):
        if not self.dataset:
            raise RuntimeError("Dataset is not initialized. Call prepare_dataset() first.")

        augmented_data = []
        for _ in range(num_samples):
            for (templ, names) in self.dataset.enumerate_comp(self.replace_ratio):
                if not templ:
                    continue
                augmented_input, _ = self.dataset.realize(templ[0], names)
                if not augmented_input:
                    continue
                augmented_data.append((" ".join(augmented_input), templ[1]))
        return augmented_data

    def save_augmented_data(self, augmented_data, original_data):
        df_aug = pd.DataFrame(augmented_data, columns=['text', 'label'])
        df_orig = pd.read_csv(self.input_csv, encoding='utf-8')[['text', 'label']]
        df_combined = pd.concat([df_orig, df_aug], ignore_index=True)
        df_combined.to_csv(self.output_csv, index=False)

    def process(self, num_samples=10):
        df = self.load_data()
        self.prepare_dataset(df)
        augmented_data = self.augment_data(num_samples)
        self.save_augmented_data(augmented_data, df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GECA")
    parser.add_argument("--dataset", type=str, required=True,
                        help="CSV file (e.g. CR.csv)")
    parser.add_argument("-naug", "--naug", type=int, default=8,
                        help="number of augmentation")
    args = parser.parse_args()

    if os.path.basename(args.dataset) == args.dataset:
        dataset_path = os.path.join("dataset", args.dataset)
    else:
        dataset_path = args.dataset

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    base = os.path.splitext(os.path.basename(dataset_path))[0]
    output_file = f"{base}_geca.csv"

    processor = GECAProcessor(dataset_path, output_file, replace_ratio=0.15)
    processor.process(num_samples=args.naug)
