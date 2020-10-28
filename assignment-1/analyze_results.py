import argparse
import os
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--target_dir", default="results", type=str, help="Where to put results...")
parser.add_argument("--res_file", default="res.csv", type=str, help="csv with results...")
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    res_csv = os.path.join(args.target_dir, args.res_file)
    # dataset, experiment, mess_prob, iteration, sum, min, max, avg
    df = pd.read_csv(res_csv)
    df.groupby(by="dataset")
    df.set_index(["dataset", "experiment", "mess_prob"])
    print(df.head())