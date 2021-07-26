import argparse
import os
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--pseudo-label-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transcriptions = {}

    with open(args.pseudo_label_dir, "r") as f:
        for line in f.readlines():
            hyp, idx = line.split(" (None-")
            idx = int(idx.replace(")\n", ""))

            transcriptions[idx] = hyp

    with open(args.tsv, "r") as tsv, open(
        os.path.join(args.output_dir, args.output_name + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.output_dir, args.output_name + ".wrd"), "w"
    ) as wrd_out:
        for i in tqdm(range(len(tsv.readlines())-1)):
            print(transcriptions[i], file=wrd_out)
            print(
                " ".join(list(transcriptions[i].replace(" ", "|"))) + " |",
                file=ltr_out,
            )


if __name__ == "__main__":
    main()
