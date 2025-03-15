import random
import csv
import argparse
import os

random.seed(0)

PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
PUNC_RATIO = 0.3

def insert_punctuation_marks(sentence, punc_ratio=PUNC_RATIO):
    words = sentence.split(' ')
    new_line = []
    q = random.randint(1, int(punc_ratio * len(words) + 1))
    qs = random.sample(range(0, len(words)), q)

    for j, word in enumerate(words):
        if j in qs:
            new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS) - 1)])
            new_line.append(word)
        else:
            new_line.append(word)
    new_line = ' '.join(new_line)
    return new_line

def main():
    parser = argparse.ArgumentParser(description="AEDA")
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
    output_file = f"{base}_aeda.csv"

    data = []
    label = []
    new_data = []

    with open(dataset_path, 'r', encoding='Windows-1252') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            data.append(row[0])
            label.append(row[1])
            for _ in range(args.naug):
                sentence_aug = insert_punctuation_marks(row[0])
                new_data.append([sentence_aug, row[1]])

    with open(output_file, "w", newline='', encoding='Windows-1252') as f:
        writer = csv.writer(f)
        writer.writerow(['sentence', 'label'])
        for i in range(len(data)):
            if len(data[i].strip()) > 0:
                writer.writerow([data[i], label[i]])
        for entry in new_data:
            writer.writerow(entry)

    print(f"Augmented Datasets as '{output_file}'.")

if __name__ == "__main__":
    main()
