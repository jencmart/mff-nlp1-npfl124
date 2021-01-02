import errno
import os
import numpy as np
from utils import *


def init_bigram_counts(word_ID_dict, data_list):
    bi_cnt = np.zeros((len(word_ID_dict), len(word_ID_dict)))
    for idx, first_word in enumerate(data_list):
        if idx + 1 == len(data_list): break
        second_word = data_list[idx + 1]
        idx1 = word_ID_dict[first_word]
        idx2 = word_ID_dict[second_word]
        bi_cnt[idx1, idx2] += 1

    return bi_cnt


def compute_qk(c_x1x2, c_x1, c_x2, n):
    if c_x1x2 == 0:
        return 0
    return (c_x1x2 / n) * np.log2((c_x1x2 * n) / (c_x1 * c_x2))


def compute_adds(ck, ck_l, ck_r, a, b, n):
    adds = 0
    # q(L , a+b)
    for left in range(ck.shape[0]):
        if left != a and left != b:
            adds += compute_qk(ck[left, a] + ck[left, b], ck_l[left], ck_r[a] + ck_r[b], n)
    # q(a+b, R)
    for right in range(ck.shape[1]):
        if right != a and right != b:
            adds += compute_qk(ck[a, right] + ck[b, right], ck_l[a] + ck_l[b], ck_r[right], n)
    # q(a+b, a+b)
    adds += compute_qk(ck[a, a] + ck[a, b] + ck[b, b] + ck[b, a], ck_l[a] + ck_l[b], ck_r[a] + ck_r[b], n)
    return adds


class Classes:
    def __init__(self, initial_word_clsID, ignored_words):
        # always the same ...
        self.ignored_classes = set([initial_word_clsID[w] for w in ignored_words])

        # changes (I will add merged to this ...)
        self.clsID_words = {v: {k} for k, v in initial_word_clsID.items()}  # 0:(w5,w11,w2)

        # changes (I will add merged to this ...)
        self.not_ignored_classes = [c for c in self.clsID_words.keys() if c not in self.ignored_classes]

        self.merged_ids = set()
        self.history = []

    # for iteration...
    def getNotIgnoredClasses(self):
        return self.not_ignored_classes

    # merge B to A
    def _merge_classes(self, clsID_a, clsID_b):
        # remove B from not ignored
        self.not_ignored_classes.remove(clsID_b)

        # add all from B to A
        self.clsID_words[clsID_a] = self.clsID_words[clsID_a].union(self.clsID_words[clsID_b])
        self.clsID_words.pop(clsID_b)
        # print("after merge: ", self.clsID_words[clsID_a], sep=" ")

        # same that this id is merged...
        self.merged_ids.add(clsID_b)

    # for "history"
    def _get_words(self, classID):
        return self.clsID_words[classID]

    def save_history_and_merge(self, min_loss, best_a, best_b):
        self.history.append((min_loss, self._get_words(best_a), self._get_words(best_b)))
        print("{:.10f}".format(min_loss), self._get_words(best_a), self._get_words(best_b), sep="\t")
        self._merge_classes(best_a, best_b)

    def getMergedIDs(self):
        return self.merged_ids


def initialize(data, ignored):
    # n: lexicon size ----------------------->>> O(n log n)
    initial_word_clsID = {w: i for i, w in enumerate(sorted(list(set(data))))}  # w1:0 w2:1 w3:2
    WC = Classes(initial_word_clsID, ignored)
    print("lexicon size (without ignored): {}".format(len(initial_word_clsID) - len(WC.ignored_classes)))

    n = len(data) - 1  # without <S>

    # 1. ck: bi counts ----------------------->>> O(n)
    ck = init_bigram_counts(initial_word_clsID, data)

    # 2. ck_l, cr_l: uni counts -------------->>> O(n)
    ck_l = np.sum(ck, axis=1)
    ck_r = np.sum(ck, axis=0)

    # 3. qk: elements of sum of I(D,E) ------>>> O(n^2)
    qk = np.zeros(ck.shape)
    for a in range(ck.shape[0]):
        for b in range(ck.shape[1]):
            qk[a, b] = compute_qk(ck[a, b], ck_l[a], ck_r[b], n)

    # 4. sk: substitutions ------------------>>> O(n)
    sk = np.sum(qk, axis=0) + np.sum(qk, axis=1) - np.diag(qk)

    # test ...
    # assert np.isclose(np.sum(qk), 4.99726326162518)
    # assert sk.shape[0] == qk.shape[0] == qk.shape[1] == ck.shape[0] == ck.shape[1] == ck_l.shape[0] == ck_r.shape[0]

    # 5. lk: losses --------------------------->>> O(n^3)
    lk = np.ones(ck.shape)  # upper triangle (without diagonal)
    min_loss, best_a, best_b = None, None, None
    for idx, a in enumerate(WC.getNotIgnoredClasses()):
        for b in WC.getNotIgnoredClasses()[idx + 1:]:  # >>> adds == O(n)
            lk[a, b] = sk[a] + sk[b] - qk[a, b] - qk[b, a] - compute_adds(ck, ck_l, ck_r, a, b, n)
            if min_loss is None or lk[a, b] < min_loss:
                min_loss, best_a, best_b = lk[a, b], a, b
    # assert np.isclose(min_loss, 0.00219656653357569)

    # 6. save first history
    WC.save_history_and_merge(min_loss, best_a, best_b)

    return ck, ck_l, ck_r, qk, sk, lk, best_a, best_b, WC, n


def compute_new_qk(qk, ck, ck_l, ck_r, N, best_b):
    qk_new = np.zeros(qk.shape)
    for a in range(ck.shape[0]):
        for b in range(ck.shape[1]):
            if a != best_b and b != best_b:
                qk_new[a, b] = compute_qk(ck[a, b], ck_l[a], ck_r[b], N)
            else:
                qk_new[a, b] = 0  # qk[a, b]  # ??? todo .. 0 ?  # 0  #
    return qk_new


def compute_classes_greedy(data, ignored, stop_at):
    # 1.-6. ----------------------------------->>> O(n^3)
    ck, ck_l, ck_r, qk, sk, lk, best_a, best_b, WC, N = initialize(data, ignored)
    # pickle.dump((ck, ck_l, ck_r, qk, sk, lk, best_a, best_b, WC, N), open("save.p", "wb"))
    # ck, ck_l, ck_r, qk, sk, lk, best_a, best_b, WC, N = pickle.load(open("save.p", "rb"))
    print("...init done...")

    while True:
        remaining = len(WC.getNotIgnoredClasses())
        print("remaining: {}".format(remaining - stop_at))
        if remaining == 1 or stop_at is not None and stop_at == remaining:
            break

        # 6.5 - Update counts ------------------->>> O(n)
        ck_1, ck_l_1, ck_r_1 = ck, ck_l, ck_r

        ck_1[best_a, :] = ck[best_a, :] + ck[best_b, :]
        ck_1[:, best_a] = ck[:, best_a] + ck[:, best_b]
        ck_1[best_b, :], ck_1[:, best_b] = 0, 0
        ck_1[list(WC.merged_ids), :], ck_1[:, list(WC.merged_ids)] = 0, 0   # just safety fallback
        ck_l_1, ck_r_1 = np.sum(ck, axis=1), np.sum(ck, axis=0)

        # 7. update qk to q_{k-1}  -------------->>> O(n^2)
        qk_1 = compute_new_qk(qk, ck_1, ck_l_1, ck_r_1, N, best_b)

        # 8. get s_{k-1} by updating sk(i) for all i != b
        sk_1 = np.sum(qk_1, axis=0) + np.sum(qk_1, axis=1) - np.diag(qk_1)

        # just safety fallback
        sk_1[list(WC.merged_ids)] = 0
        qk_1[list(WC.merged_ids), :] = 0
        qk_1[:, list(WC.merged_ids)] = 0

        # 9., 10. update losses + save minimal loss of MI and new best_a best_b
        lk_1 = np.ones(lk.shape)  # upper triangle (without diagonal)
        old_best_a, old_best_b, min_loss, best_a, best_b = best_a, best_b, None, None, None
        for idx, i in enumerate(WC.getNotIgnoredClasses()):
            for j in WC.getNotIgnoredClasses()[idx + 1:]:  # >>> adds == O(n)
                if i == old_best_a or j == old_best_a:
                    # difference in sk ... substitutions
                    lk_1[i, j] = lk[i, j] - sk[i] + sk_1[i] - sk[j] + sk_1[j]

                    # Difference in additions? ... # q(L , a+b)  q(a+b, R) q(a+b,a+b)
                    # + qk(i+j,a)
                    lk_1[i, j] += compute_qk(ck[i, old_best_a] + ck[j, old_best_a], ck_l[i] + ck_l[j], ck_r[old_best_a], N)
                    # + qk(a, i+j)
                    lk_1[i, j] += compute_qk(ck[old_best_a, i] + ck[old_best_a, j], ck_l[old_best_a], ck_r[i] + ck_r[j], N)
                    # + qk(i+j, b)
                    lk_1[i, j] += compute_qk(ck[i, old_best_b] + ck[j, old_best_b], ck_l[i] + ck_l[j], ck_r[old_best_b], N)
                    # + qk(b, i+j)
                    lk_1[i, j] += compute_qk(ck[old_best_b, i] + ck[old_best_b, j], ck_l[old_best_b], ck_r[i] + ck_r[j], N)

                    # intersection for sk ?
                    # - qk_1(i+j, a)
                    lk_1[i, j] -= compute_qk(ck_1[i, old_best_a] + ck_1[j, old_best_a], ck_l_1[i] + ck_l_1[j], ck_r_1[old_best_a], N)
                    # - qk_1(a, i+j)
                    lk_1[i, j] -= compute_qk(ck_1[old_best_a, i] + ck_1[old_best_a, j], ck_l_1[old_best_a], ck_r_1[i] + ck_r_1[j], N)

                else:
                    lk_1[i, j] = lk[i, j]
                if min_loss is None or lk_1[i, j] < min_loss:
                    min_loss, best_a, best_b = lk_1[i, j], i, j

        # 11. save best_a, best_b to HISTORY
        WC.save_history_and_merge(min_loss, best_a, best_b)

        # 12. qk sk lk = new ones
        qk, sk, lk = qk_1, sk_1, lk_1
        ck, ck_l, ck_r = ck_1, ck_l_1, ck_r_1
        # I = np.sum(qk)
        # print(I)
    return WC


def save_merge_history(wc_obj, fname):
    # Save Results (create directory if not exist)
    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(fname, "w") as f:
        f.write("loss\tcls_a\tcls_b\n")
        for x in wc_obj.history:
            # print(x)
            loss, set_a, set_b = x
            str_loss = "{:.10f}".format(loss)
            str_set_a = set_a.__str__()
            str_set_b = set_b.__str__()
            f.write(str_loss + "\t" + str_set_a + "\t" + str_set_b + "\n")


def save_words_with_classes(wc_obj, fname):
    # print out all the members of your 15 classes and attach them too. (meaning on all the data??)

    # Save Results (create directory if not exist)
    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(fname, "w") as f:
        for cls_ID, words_set in wc_obj.clsID_words.items():
            # skip initially ignored classes ( aka. words with not enough occurrences)
            if cls_ID not in wc_obj.ignored_classes:
                str_set = sorted(list(words_set)).__str__()
                f.write(str_set + "\n")


def filter_occurs(data, min_occurs):  # 2 51
    counts = CountDict()
    for word in data:
        counts.insert(word)
    result = []
    ignored = set([])
    for word in data:
        # if counts._dict[word] >= min_occurs:
        result.append(word)
        if counts._dict[word] < min_occurs:
            ignored.add(word)
    return result, ignored


if __name__ == "__main__":
    # DATASETS
    dataset_dir = "dataset"
    datasets = {
        "en": "TEXTEN1.ptg",
        "cz": "TEXTCZ1.ptg"
                }
    # >>> encode CZ dataset to UTF >>>
    path = os.path.join(dataset_dir, datasets["cz"] + "-utf")
    if not os.path.isfile(path):
        encode_as_utf(os.path.join(dataset_dir, datasets["cz"]))
    datasets["cz"] = datasets["cz"] + "-utf"
    # <<< encode CZ dataset to UTF <<<

    # ... On each dataset ...
    experiments = [
        {"name": "words", "min_occurs": 10, "cnt_data": 8000, "cnt_classes_stop": 1},
        {"name": "words", "min_occurs": 10, "cnt_data": 8000, "cnt_classes_stop": 15, "words_to_classes": True},
        {"name": "tags", "min_occurs": 5, "cnt_data": None, "cnt_classes_stop": 1},
    ]
    for experiment in experiments:
        for lang, file_name in datasets.items():
            print("Dataset: {}".format(file_name))
            part = 0 if experiment['name'] == "words" else 1
            dataset, _, _ = load_dataset(os.path.join(dataset_dir, file_name), part=part)

            # take first 8,000
            dataset = dataset[:experiment['cnt_data']] if experiment['cnt_data'] is not None else dataset

            # occurring x-times in 8k !!
            _, ignored = filter_occurs(dataset, experiment['min_occurs'])

            extra_symb = "<S>"
            dataset.insert(0, extra_symb)
            ignored.add(extra_symb)
            word_cls_obj = compute_classes_greedy(dataset, ignored, stop_at=experiment['cnt_classes_stop'])

            filename = "word_results/" + "merge_history_" + lang + "_" + experiment["name"] + ".csv"
            save_merge_history(word_cls_obj, filename)

            if 'words_to_classes' in experiment:
                filename = "word_results/" + "words_in_classes_"  + lang + "_" + experiment["name"] + "_cls:" + str(experiment['cnt_classes_stop'])  + ".csv"
                save_words_with_classes(word_cls_obj, filename)
