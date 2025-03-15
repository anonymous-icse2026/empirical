import argparse
import random
import subprocess
import os
import tempfile
import pandas as pd
import shutil


def generate_input_txt(data_csv, input_txt_path):
    df = pd.read_csv(data_csv)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV needs 'text', 'label'")

    label_groups = {}
    for label, group in df.groupby('label'):
        label_groups[label] = group['text'].tolist()

    with open(input_txt_path, 'w', encoding='utf-8') as fout:
        for _, row in df.iterrows():
            sentence1 = row['text']
            label = row['label']
            candidates = label_groups[label]
            if len(candidates) == 1:
                sentence2 = candidates[0]
            else:
                filtered = [s for s in candidates if s != sentence1]
                sentence2 = random.choice(filtered) if filtered else sentence1
            fout.write(f"{sentence1}\t{sentence2}\t{label}\n")


def main():
    parser = argparse.ArgumentParser(description="augment script")
    parser.add_argument("-da", "--da", type=str, required=True,
                        help="data augmentation identifier (e.g. stg, eda, ts, aeda, geca)")
    parser.add_argument("--dataset", type=str, help="csv file (e.g. CR.csv)")
    parser.add_argument("-naug", "--naug", type=int, default=1, help="number of augmentation")
    parser.add_argument('-s', '--save_file', type=str, help="model checkpoint file (stg)")
    parser.add_argument('-v', '--vocab_file', type=str, help="vocab (stg)")
    parser.add_argument('-bs', '--beam_size', type=int, help="beam search (stg)")
    parser.add_argument('-d', '--data_file', type=str, help="input csv (e.g. CR.csv)")
    parser.add_argument('-o', '--output_file', type=str, help="output csv (stg)")
    args = parser.parse_args()

    if args.data_file is None:
        if args.dataset is not None:
            args.data_file = os.path.join("dataset", args.dataset)
        else:
            raise ValueError("We need datasets")

    if args.da == "stg":
        defaults = {
            "save_file": "stg/model.ckpt",
            "vocab_file": "stg/vocab.pkl",
            "beam_size": 10
        }
        if args.save_file is None:
            args.save_file = defaults["save_file"]
        if args.vocab_file is None:
            args.vocab_file = defaults["vocab_file"]
        if args.beam_size is None:
            args.beam_size = defaults["beam_size"]
        script_name = "generate.py"
        generate_script_path = os.path.join(args.da, script_name)

        if args.output_file is None:
            base = os.path.splitext(os.path.basename(args.data_file))[0]
            args.output_file = f"{base}_{args.da}.csv"

        temp_input = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', encoding='utf-8')
        temp_input_path = temp_input.name
        temp_input.close()
        generate_input_txt(args.data_file, temp_input_path)

        output_files = []
        for i in range(args.naug):
            temp_output = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', encoding='utf-8')
            temp_output_path = temp_output.name
            temp_output.close()
            output_files.append(temp_output_path)

            command = ["python", generate_script_path, "-i", temp_input_path, "-o", temp_output_path]
            command.extend([
                "-s", os.path.abspath(args.save_file),
                "-v", os.path.abspath(args.vocab_file),
                "-bs", str(args.beam_size)
            ])
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"{generate_script_path} Error:", e)
                for f in output_files:
                    if os.path.exists(f):
                        os.remove(f)
                os.remove(temp_input_path)
                raise e

        df_list = []
        for file in output_files:
            try:
                with open(file, 'r', encoding='utf-8', errors='replace') as fin:
                    df = pd.read_csv(fin, sep='\t', header=None, names=['text', 'label'])
                    df_list.append(df)
            except Exception as e:
                print(f"{file} Fail to read:", e)
        aug_df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame(columns=['text', 'label'])

        orig_df = pd.read_csv(args.data_file)
        final_df = pd.concat([orig_df, aug_df], ignore_index=True)
        final_df.to_csv(args.output_file, index=False, encoding='utf-8')
        print(f"Augmented Datasets as {args.output_file}")

        os.remove(temp_input_path)
        for file in output_files:
            if os.path.exists(file):
                os.remove(file)


    elif args.da in ["eda", "ts", "aeda", "geca", "ca"]:
        script_name = f"{args.da}.py"
        generate_script_path = os.path.join(args.da, script_name)
        command = [
            "python", generate_script_path,
            "--dataset", os.path.abspath(args.data_file),
            "-naug", str(args.naug)
        ]
        #print(f"[{args.da.upper()}] aug.. : {generate_script_path} {' '.join(command)}")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"{generate_script_path} Error:", e)
            raise e
        base = os.path.splitext(os.path.basename(args.data_file))[0]
        expected_output = f"dataset/{base}_{args.da}.csv"
        if os.path.exists(expected_output):
            new_path = os.path.join("dataset", expected_output)
            shutil.move(expected_output, new_path)
    else:
        raise ValueError(f"Unknown identifier: {args.da}")


if __name__ == "__main__":
    main()
