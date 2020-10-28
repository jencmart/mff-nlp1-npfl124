import argparse
import math
import os
import random
import pandas as pd


def encode_as_utf(filename="dataset/TEXTCZ1.txt", endcoding="iso-8859-2"):
    with open(filename, "rt", encoding=endcoding) as f:
        lines = f.readlines()

    with open(filename + "-utf", "w") as f:
        for line in lines:
            f.write(line)


def load_dataset(file_path):
    with open(file_path) as f:
        word_list = f.readlines()
        word_list = [w.strip() for w in word_list]
        lexicon = sorted(list(set(word_list)))
        charset =  list(set([ c for c in "".join(word_list)]))
        return word_list, lexicon, charset


def compute_conditional_probability(w1_wm_probs, joint_probs, delim):
    """
    We want to estimate P(w_n | w_1, w_2 .. w_{n-1}) = P(w_1, w_2 .. w_{n-1}, w_n) / P(w_1, w_2 .. w_{n-1})
    :param w1_wm_probs: dictionary { "w_1<DELIM>w_2<DELIM>...w_{n-1}" : P(w_1, w_2 .. w_{n-1}), }
    :param joint_probs:  dictionary { "w_1<DELIM>...w_{n-1}<DELIM>w_n" : P(w_1, w_2 .. w_{n-1}, w_n), }
    :param delim: <DELIM> of the word in key of dicitonary
    :return:
    """
    prob_of_wn_given_I = {}

    def split_w1_wm_n(w1_wm_wn, delim):
        wn = w1_wm_wn.split(delim)[-1]
        w1_wm = w1_wm_wn[:-1 * (len(wn) + len(delim))]
        return w1_wm, wn

    # P(wn | w1 .. wm) = P(w1...wm.wm) / P(w1..wm)
    for w1_wm_wn, P_w1_wm_wn in joint_probs.items():
        w1_wm, wn = split_w1_wm_n(w1_wm_wn, delim)
        P_w1_wm = w1_wm_probs[w1_wm]

        # if w1_wm == "LENKA" or w1_wm == "LINDAUROVÁ" :
        #     print("P( {} | {} ) = {}".format(wn, w1_wm, P_w1_wm_wn / P_w1_wm))
        if wn in prob_of_wn_given_I:
            prob_of_wn_given_I[wn][w1_wm] = P_w1_wm_wn / P_w1_wm
        else:
            prob_of_wn_given_I[wn] = {w1_wm: P_w1_wm_wn / P_w1_wm}

    return prob_of_wn_given_I


def compute_counts(dataset_list, n, delim, end_op="<END>"):
    dic = {}
    # count = len(dataset_list) - n + 1
    for i in range(len(dataset_list)):
        wrd = ""
        for j in range(n):
            if i+j >=len(dataset_list):
                wrd += end_op
            else:
                wrd += dataset_list[i+j]
            if j+1 < n:
                wrd += delim
        if wrd in dic:
            dic[wrd] += 1
        else:
            dic[wrd] = 1
    # done here for better numerical stability
    for k, v in dic.items():
        dic[k] = v/len(dataset_list)

    return dic


def compute_conditional_entropy(prob_of_wn_given_I, w1_wm_wn_probs, delim):
    """
    :param prob_of_wn_given_I: dictionary { wn : { w1<DELIM>...w_M : P(wn | w_1, w_2 .. w_{n-1}) , ... }, ... }
    :param w1_wm_wn_probs:  dictionary { "w_1<DELIM>...w_{n-1}<DELIM>w_n" : P(w_1, w_2 .. w_{n-1}, w_n), }
    :param delim: <DELIM> of the word in key of dicitonary
    :return:
    """
    H_wn_given_I = {}
    # H(wn | w1 .. wm) = P(w1...wm.wm) * log (P(wn | w1...wn))
    for wn, probs_wn_given_w1_wm in prob_of_wn_given_I.items():
        H_wn_given_I[wn] = 0
        for w1_wm, prob_wn_given_w1_wm in probs_wn_given_w1_wm.items():
            prob_w1_wm_wn = w1_wm_wn_probs[w1_wm + delim + wn]
            H_wn_given_I[wn] += prob_w1_wm_wn * math.log2(prob_wn_given_w1_wm)
        H_wn_given_I[wn] *= -1
    return H_wn_given_I


def compute_entropy_on_dataset(dataset, n, delim):
    # count prob .. p(w1) = count(w1) / |dataset|
    wrd_probs = compute_counts(dataset_list=dataset, n=n, delim=delim)
    # count prob .. p(w1, w2) = count(w1 w2) / |dataset - 1|
    joint_probs = compute_counts(dataset_list=dataset, n=n + 1, delim=delim)  # n+1 words ...
    # compute conditional prob  p(w2 | w1) = p(w1, w2) / p(w1)
    prob_of_wn_given_I = compute_conditional_probability(w1_wm_probs=wrd_probs, joint_probs=joint_probs, delim=delim)
    # Compute conditional entropy of J ... H(J|I)
    cond_entrop = compute_conditional_entropy(prob_of_wn_given_I, joint_probs, delim)

    return cond_entrop


def char_mod(data, _, charset, prob):
    new_dataset = []
    for i, w in enumerate(data):
        w_new = ""
        for c in w:
            if random.random() < prob:
                w_new += random.choice(charset)
            else:
                w_new += c
        new_dataset.append(w_new)
    return new_dataset


def word_mod(data, lexicon, _, prob):
    new_dataset = []
    for i, w in enumerate(data):
        if random.random() < prob:
            new_dataset.append(random.choice(lexicon))
        else:
            new_dataset.append(w)
    return new_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default="dataset", type=str, help="Path to the dataset.")
parser.add_argument("--experiment_repeats", default=10, type=int, help="How many times to repeat each experiment...")
parser.add_argument("--n", default=1, type=int, help="On how many previous words would you like to condition...")
parser.add_argument("--delim", default=" ", type=str, help="What delimiter to distinct words...")
parser.add_argument("--target_dir", default="results", type=str, help="Where to put results...")
parser.add_argument("--res_file", default="res.csv", type=str, help="csv with results...")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # DATASETS
    datasets = {"cz": "TEXTCZ1.txt", "en": "TEXTEN1.txt"}
    # >>> encode CZ dataset to UTF >>>
    path = os.path.join(args.dataset_dir, datasets["cz"] + "-utf")
    if not os.path.isfile(path):
        encode_as_utf(os.path.join(args.dataset_dir, datasets["cz"]))
    datasets["cz"] = datasets["cz"] + "-utf"
    # <<< encode CZ dataset to UTF <<<

    # EXPERIMENTS
    experiments = [{"name": "char-modif", "mess_probs": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1], "mess-f": char_mod},
                   {"name": "word-modif", "mess_probs": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1], "mess-f": word_mod}]

    res_csv = os.path.join(args.target_dir, args.res_file)
    if os.path.isfile(res_csv):
        os.remove(res_csv)

    # ... On each dataset ...
    for lang, fname in datasets.items():
        print("Dataset: {}".format(fname))
        dataset, lexicon, charset = load_dataset(os.path.join(args.dataset_dir, fname))

        # ... Perform each experiment ...
        for experiment in experiments:
            for prob in experiment['mess_probs']:
                print("\tExperiment: {}, Mess: {}\n\t\tRepeat:".format(experiment['name'], prob), end="")

                # ... Repeat it given number of times ...
                dict_10_run_avg = {}
                for repeat_idx in range(args.experiment_repeats):
                    print("{},".format(repeat_idx), end="")
                    # ... and result of the experiment ...
                    modified_dataset = experiment['mess-f'](dataset, lexicon, charset, prob)
                    dic = compute_entropy_on_dataset(modified_dataset, args.n, args.delim)
                    for k, v in dic.items():
                        if k in dict_10_run_avg:
                            dict_10_run_avg[k] = (dict_10_run_avg[k][0]+v, dict_10_run_avg[k][1] + 1)
                        else:
                            dict_10_run_avg[k] = (v, 1)

                # ... And save it to file ...
                if not os.path.isdir(args.target_dir):
                    os.mkdir(args.target_dir)
                if not os.path.isfile(res_csv):
                    with open(res_csv, "w") as f:
                        f.write("dataset,experiment,mess_prob,word,avg_1\n")
                with open(res_csv, "a") as f:
                    for k, v in dict_10_run_avg.items():
                        if k == '"':
                            k = '""""'
                        else:
                            k = k.replace('"', "'")
                            k = '"' + k + '"'
                        f.write("{},{},{},{},{}\n".format(lang,
                                                             experiment['name'],
                                                             prob,
                                                             k, v[0]/args.experiment_repeats))
                print("OK")

