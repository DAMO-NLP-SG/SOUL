import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, required=True, help="path of input")
    return parser.parse_args()

def main():
    args = parse_args()

    input_path = args.path
    df = pd.read_csv(input_path)

    pred = [str(i).lower().strip() for i in df["prediction"]]
    true = [str(i).lower().strip() for i in df["label"]]

    correct = df[df["prediction"] == df["label"]]
    print(input_path, f"{len(correct)}/{len(df)}", accuracy_score(true, pred))
    print(classification_report(true, pred, digits=4, zero_division=0))

if __name__ == "__main__":
    main()