import errno
import os
import numpy as np
from utils import *


def compute_counts(dataset, min_dist=1, max_dist=1, reverse=False):  # 2 51
    bigram_count_dict = CountDict()
    unigram_count_dict = CountDict()
    max_index = len(dataset) - 1
    for idx, first_word in enumerate(dataset):
        unigram_count_dict.insert(first_word)
        start_pos = min(max_index, idx + min_dist)
        end_pos = min(max_index, idx + max_dist + 1)  # +1 because we want 'max_dist' inclusive
        for next_idx in range(start_pos, end_pos):
            second_word = dataset[next_idx]
            key = first_word + " " + second_word
            bigram_count_dict.insert(key)
        if reverse:
            start_pos = max(0, idx - min_dist + 1)
            end_pos = max(0, idx - max_dist)  # +1 because we want 'max_dist' inclusive
            for next_idx in range(end_pos, start_pos):
                second_word = dataset[next_idx]
                key = first_word + " " + second_word
                bigram_count_dict.insert(key)

    bigram_freq = bigram_count_dict.get_frequencies(len(dataset))
    unigram_freq = unigram_count_dict.get_frequencies(len(dataset), limit=10)
    return bigram_freq, unigram_freq


def compute_pointwise_mu(e, data, reverse=False):
    joint_probs, single_probs = compute_counts(data, min_dist=e["min_dist"], max_dist=e["max_dist"], reverse=reverse)
    result = {}
    for key_joint, p_joint in joint_probs.items():
        x1, x2 = key_joint.split(" ")
        p_x1 = single_probs[x1]
        p_x2 = single_probs[x2]
        if p_x1 is not None and p_x2 is not None:
            result[key_joint] = np.log2(p_joint / (p_x1 * p_x2))
    result = [(k, v) for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)]
    return result


def save_result(res, fname, max_lines=20):
    # Save Results (create directory if not exist)
    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(fname, "w") as f:
        f.write("x1,x2,pmu\n")
        for x1x2, v in res:
            max_lines -= 1
            x1, x2 = x1x2.split(" ")
            f.write(x1 + "," + x2 + "," + str(v) + "\n")
            if max_lines == 0:
                break


if __name__ == "__main__":
    # DATASETS
    dataset_dir = "dataset"
    datasets = {"cz": "TEXTCZ1.txt", "en": "TEXTEN1.txt"}

    # >>> encode CZ dataset to UTF >>>
    path = os.path.join(dataset_dir, datasets["cz"] + "-utf")
    if not os.path.isfile(path):
        encode_as_utf(os.path.join(dataset_dir, datasets["cz"]))
    datasets["cz"] = datasets["cz"] + "-utf"
    # <<< encode CZ dataset to UTF <<<

    # ... On each dataset ...
    experiments = [
        {"name": "close", "min_dist": 1, "max_dist": 1, "backward": False},
        {"name": "far", "min_dist": 1+1, "max_dist": 50+1, "backward": True},
    ]
    for experiment in experiments:
        for lang, fname in datasets.items():
            print("Dataset: {}".format(fname))
            dataset, _, _ = load_dataset(os.path.join(dataset_dir, fname))

            result = compute_pointwise_mu(experiment, dataset, reverse=False)
            filename = "friend_results/"+"best_friends_"+lang+"_"+experiment["name"]+".csv"
            save_result(result, filename)

            if experiment["backward"]:
                result = compute_pointwise_mu(experiment, dataset, reverse=True)
                filename = "friend_results/" + "best_friends_" + lang + "_" + experiment["name"] + "_reversed" + ".csv"
                save_result(result, filename)
