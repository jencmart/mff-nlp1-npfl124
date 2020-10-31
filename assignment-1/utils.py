import errno
import os


def save_line_to_csv(filename, string, csv_vals):
    # Create directory if not exist
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    # Create CSV header ...
    if not os.path.isfile(filename):
        with open(filename, "w") as f:
            f.write(csv_vals + "\n")
    # Save
    with open(filename, "a") as f:
        f.write(string + "\n")


def encode_as_utf(filename="dataset/TEXTCZ1.txt", endcoding="iso-8859-2"):
    with open(filename, "rt", encoding=endcoding) as f:
        lines = f.readlines()

    with open(filename + "-utf", "w") as f:
        for line in lines:
            f.write(line)


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def load_dataset(file_path, frm=None, to=None):
    if frm is None:
        frm = 0
    if to is None:
        to = file_len(file_path)

    with open(file_path) as f:
        word_list = f.readlines()
        word_list = [w.strip() for w in word_list]
        word_list = word_list[frm:to]
        lexicon = sorted(list(set(word_list)))
        charset = list(set([c for c in "".join(word_list)]))
        return word_list, lexicon, charset


def compute_conditional_probability(w1_wm_probs, w1_wm_wn_probs, delim):
    """
    We want to estimate P(w_n | w_1, w_2 .. w_{n-1}) = P(w_1, w_2 .. w_{n-1}, w_n) / P(w_1, w_2 .. w_{n-1})
    :param w1_wm_probs: dictionary { "w_1<DELIM>w_2<DELIM>...w_{n-1}" : P(w_1, w_2 .. w_{n-1}), }
    :param w1_wm_wn_probs:  dictionary { "w_1<DELIM>...w_{n-1}<DELIM>w_n" : P(w_1, w_2 .. w_{n-1}, w_n), }
    :param delim: <DELIM> of the word in key of dicitonary
    :return:
    """
    prob_of_wn_given_I = {}

    def split_w1_wm_n(w1_wm_wn, delim):
        wn = w1_wm_wn.split(delim)[-1]
        w1_wm = w1_wm_wn[:-1 * (len(wn) + len(delim))]
        return w1_wm, wn

    # P(wn | w1 .. wm) = P(w1...wm.wm) / P(w1..wm)
    # S ahoj
    for w1_wm_wn, P_w1_wm_wn in w1_wm_wn_probs.items():
        w1_wm, wn = split_w1_wm_n(w1_wm_wn, delim)
        # print(w1_wm_wn)
        # print(">"+w1_wm+"<")
        P_w1_wm = w1_wm_probs[w1_wm]

        # if w1_wm == "LENKA" or w1_wm == "LINDAUROVÁ" :
        #     print("P( {} | {} ) = {}".format(wn, w1_wm, P_w1_wm_wn / P_w1_wm))
        if wn in prob_of_wn_given_I:
            prob_of_wn_given_I[wn][w1_wm] = P_w1_wm_wn / P_w1_wm
        else:
            prob_of_wn_given_I[wn] = {w1_wm: P_w1_wm_wn / P_w1_wm}

    return prob_of_wn_given_I


def compute_ngram(data, n, delim=" ", start_op="<S>"):
    assert n >= 0, "n must be >=0"

    if n == 0:
        lex = sorted(list(set(data)))
        return {w: 1 / len(lex) for w in data}
    if n == 1:
        return compute_counts(data, n=1, delim=delim, start_op=start_op, max_n=1)

    counts_w1_wm = compute_counts(data, n=n-1, delim=delim, start_op=start_op, max_n=n)
    counts_w1_wm_wn = compute_counts(data, n=n, delim=delim, start_op=start_op, max_n=n)
    return compute_conditional_probability(counts_w1_wm, counts_w1_wm_wn, delim)


def compute_counts(data, n, delim=" ", start_op="<S>", max_n=1):
    dic = {}
    dataset_list = data.copy()
    # Add the starts ... <S>
    # max_n represents the n for the resulting n-gram model...
    for i in range(max_n - 1):
        dataset_list.insert(0, start_op)
    data_len = len(dataset_list)

    for i in range(data_len - max_n + 1):
        wrd = ""
        for j in range(n):
            wrd += dataset_list[i+j]
            if j+1 < n:
                wrd += delim
        if wrd in dic:
            dic[wrd] += 1
        else:
            dic[wrd] = 1
    # done here for better numerical stability
    for k, v in dic.items():
        # print("{} , {}".format(k,v))
        dic[k] = v/data_len

    return dic
