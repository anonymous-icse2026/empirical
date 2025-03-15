import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Reduce dataset size")
parser.add_argument("--dataset", type=str, required=True, help="Path CSV")
args = parser.parse_args()

file_path = args.dataset
file_name = file_path.rsplit('.', 1)[0]

df = pd.read_csv(file_path, encoding='utf-8')

label_counts = df['label'].value_counts()
print("Instances for each label:")
print(label_counts)

def reduce_labels(df, label, reduction_factor=10):
    label_df = df[df['label'] == label]
    reduced_label_df = label_df.sample(frac=1/reduction_factor, random_state=42)
    return reduced_label_df

labels_to_reduce = [1, 2, 3, 4, 5]
reduced_dfs = []

for label in labels_to_reduce:
    if label in df['label'].values:
        reduced_dfs.append(reduce_labels(df, label))

remaining_df = df[~df['label'].isin(labels_to_reduce)]
reduced_df = pd.concat(reduced_dfs + [remaining_df], ignore_index=True)

output_file_path = f"{file_name}_biased.csv"
reduced_df.to_csv(output_file_path, index=False)
