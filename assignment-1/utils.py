
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
    # S ahoj
    for w1_wm_wn, P_w1_wm_wn in joint_probs.items():
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


def compute_counts(dataset_list, n, delim, start_op="<BEG>"):
    dic = {}
    data_len = len(dataset_list)

    # Add the starts ... <S>
    for i in range(n):
        dataset_list.insert(0, start_op)

    for i in range(data_len - n + 1):
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
        dic[k] = v/data_len

    return dic
