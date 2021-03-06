import argparse
import math
import os

from utils import load_dataset, encode_as_utf, file_len, save_line_to_csv, compute_ngram


def get_prob(n_gram_list, wi, words_before, uniform_prob):
    V = 1/list(uniform_prob.values())[0]
    prob = [0]*len(n_gram_list)
    bad_prob = [0]*len(n_gram_list)

    # TODO OOV word --> 1/V for all n-grams
    if wi not in uniform_prob:
        # print("\t\t<unk> [{}]".format(wi))
        return [1/V]*len(n_gram_list)

    for ii, n_gram in enumerate(n_gram_list):
        # TODO Dealing with zero probability
        # 0-grams, 1-grams ... here always correct
        # 2-grams ... will be set to 1/V (and then also all consequent n-grams)
        # (3+)-grams ... can be 0, if lower order n-gram have correct probability
        if ii >= 2 and words_before[ii - 2] not in n_gram[wi]:
            if ii == 2 or prob[ii-1] == 0:
                bad_prob[ii] = 1/V
            else:
                bad_prob[ii] = 0  # todo - this can be dangerous  L_3 = 1, P_3 = 0 ---> log(0)
            continue

        if ii == 0:
            prob[ii] = 1/V
        elif ii == 1:
            prob[ii] = n_gram[wi]
        else:
            prob[ii] = n_gram[wi][words_before[ii - 2]]
    return [i+j for i, j in zip(prob, bad_prob)]


def compute_smoothing_using_lm(n_gram_distribs, dataset, delim=" ", start_op="<S>", eps=0.0001):
    # 1. Init the lambdas ..
    lambdas = [1/len(n_gram_distribs)]*len(n_gram_distribs)

    # EM
    itr = 0
    while True:
        itr += 1
        expected_counts = [0]*len(lambdas)
        # 1. Expectation ...
        for i in range(len(dataset)):
            w1_wm = get_previous_words(i, len(lambdas), dataset, delim, start_op)
            prob_list = get_prob(n_gram_distribs, dataset[i], w1_wm, n_gram_distribs[0])
            p_m = sum([i*j for i, j in zip(prob_list, lambdas)])
            # update each lambda
            for N in range(len(lambdas)):
                p_n = prob_list[N]  # this can be zero ...
                # assert p_n > 0
                # p_n = p_n if 0 != p_n else 1/list(n_gram_distribs[0].values())[0]
                expected_counts[N] += (lambdas[N]*p_n) / p_m

        # 2. Maximization ..
        new_lambdas = []
        for cnt in expected_counts:
            new_l = cnt/sum(expected_counts)
            new_lambdas.append(new_l)

        # Check  if enough
        change = False
        #  >>>> Print stats
        print("\t\tIteration: {}, Changes(l_0 .. l_{}): ".format(itr, len(lambdas)), end="")
        for idx, (l, nl) in enumerate(zip(lambdas, new_lambdas)):
            print("{0:.4f}".format(nl-l), end="")
            if idx+1 < len(lambdas):
                print(", ", end="")
            if math.fabs(nl-l) > eps:
                change = True
        print()
        # <<<< Print stats
        lambdas = new_lambdas
        if not change:
            break
    return tuple(lambdas)


def compute_cross_entropy(model, lambdas, dataset, delim, start_op):
    assert len(model) == len(lambdas)
    n = len(model) - 1  # we have up to n-grams
    assert n > 1  # we want more than uni-grams .. we have 3 ...
    H = 0

    # For each word in the dataset compute smoothed probability ..
    for idx in range(len(dataset)):
        wi = dataset[idx]
        # Word w_{i-n} ... w_{i-1}
        words_before = get_previous_words(idx, n, dataset, delim, start_op)
        # p_s(w_i | w_{i-2}, w_{i-1} ) = L0*P_0 + L1*P1 + L2*P2 + L3*P3
        prob_lst = get_prob(model, wi, words_before, model[0])
        p_smooth = sum([i*j for i, j in zip(prob_lst, lambdas)])
        if p_smooth <= 0:
            print(prob_lst)
            print(lambdas)
            assert p_smooth > 0
        # Cross Entropy H(p,q) = -1 * SUM p_real * log(q_trained)
        # Shannon-McMillan-Breiman theorem ... H(p,q) ~ -1/N * SUM log(q_trained)
        H += math.log2(p_smooth)
    H = (-1 / len(dataset)) * H
    return H


def get_previous_words(idx, n, dataset, delim, start_op):
    words_before = []
    w_before = ""
    to = idx-1
    frm = idx-n
    for i in range(to, frm, -1):  # for n-gram we seek conditional prob of n-1 .. and his reversed range do it
        if i < 0:
            w_before = start_op + w_before
        else:
            w_before = dataset[i] + w_before
        words_before.append(w_before)
        if i - 1 > frm:
            w_before = delim + w_before
    return words_before


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default="dataset", type=str, help="Path to the dataset.")
parser.add_argument("--experiment_repeats", default=10, type=int, help="How many times to repeat each experiment...")
parser.add_argument("--delim", default=" ", type=str, help="What delimiter to distinct words...")
parser.add_argument("--target_dir", default="results", type=str, help="Where to put results...")
parser.add_argument("--res_file", default="cross-ent.csv", type=str, help="csv with results...")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    TEST = 20000
    HOLDOUT = 40000
    # DATASETS
    datasets = {"cz": "TEXTCZ1.txt",
                "en": "TEXTEN1.txt"}
    # >>> encode CZ dataset to UTF >>>
    path = os.path.join(args.dataset_dir, datasets["cz"] + "-utf")
    if not os.path.isfile(path):
        encode_as_utf(os.path.join(args.dataset_dir, datasets["cz"]))
    datasets["cz"] = datasets["cz"] + "-utf"
    # <<< encode CZ dataset to UTF <<<

    # EXPERIMENTS
    experiments = [{"name": "L3-inc", "val": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1]},
                   {"name": "L3-dec", "val": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1], }]

    # Clean the result dir...
    res_csv = os.path.join(args.target_dir, args.res_file)
    res_cvs_params = res_csv + ".params"
    if os.path.isfile(res_csv):
        os.remove(res_csv)
    if os.path.isfile(res_cvs_params):
        os.remove(res_cvs_params)

    # ... On each dataset ...
    for lang, fname in datasets.items():
        print("Dataset: {}".format(fname))

        # 1. Split the dataset ...
        cnt_lines = file_len(os.path.join(args.dataset_dir, fname))
        test_dataset, _, _ = load_dataset(os.path.join(args.dataset_dir, fname), frm=cnt_lines-TEST, to=cnt_lines)
        holdout_dataset, _, _ = load_dataset(os.path.join(args.dataset_dir, fname), frm=cnt_lines-TEST-HOLDOUT, to=cnt_lines-TEST)
        train_dataset, _, _ = load_dataset(os.path.join(args.dataset_dir, fname), frm=0, to=cnt_lines-TEST-HOLDOUT)
        assert len(test_dataset) + len(holdout_dataset) + len(train_dataset) == cnt_lines

        # 2. Calculate n-grams ...
        start_op = "<S>"
        N = 3  # calcualte up to 3 grams
        print("\tComputing n-grams (0-grams ... {}-grams)".format(N))
        model = [compute_ngram(data=train_dataset, n=i, delim=args.delim, start_op=start_op) for i in range(N+1)]

        # 3. Compute Good and Bad Smoothing Parameters and save them...
        epsilon = 0.0001
        # Bad lambdas
        print("\tComputing lambdas (from Train Data). EM epsilon={}".format(epsilon))
        L0_no, L1_no, L2_no, L3_no = compute_smoothing_using_lm(model, train_dataset)
        print("\tLambdas: [{},{},{},{}]".format(L0_no, L1_no, L2_no, L3_no))
        s = "{},train,{},{},{},{}".format(lang, L0_no, L1_no, L2_no, L3_no)
        save_line_to_csv(res_cvs_params, s, csv_vals="dataset,from,l0,l1,l2,l3")

        # Good lambdas
        print("\tComputing lambdas (from Holdout Data). EM epsilon={}".format(epsilon))
        L0, L1, L2, L3 = compute_smoothing_using_lm(model, holdout_dataset)
        print("\tLambdas: [{},{},{},{}]".format(L0, L1, L2, L3))
        s = "{},holdout,{},{},{},{}".format(lang, L0, L1, L2, L3)
        save_line_to_csv(res_cvs_params, s, csv_vals="dataset,from,l0,l1,l2,l3")

        # 4. Compute Cross Entropy using original parameters, save result ...
        cross_ent = compute_cross_entropy(model, [L0, L1, L2, L3], test_dataset, args.delim, start_op)
        print("\tCross-Entropy default, H(p,q) = {:.4f}".format(cross_ent))
        csv_vals="dataset,modif,modif_val,L3_val,cross_entropy"
        save_line_to_csv(res_csv, "{},{},{},{},{}".format(lang, "default", "0.0", L3, cross_ent), csv_vals)
        print("\t--------------------")

        # 5. Modify Params + Compute Cross Entropy + Save result ...
        default_lambdas = [L0, L1, L2, L3]
        modified = [0, 0, 0, 0]
        for inc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
            # Change L3
            modified[3] = L3 + (1.0 - L3) * inc  # add inc * difference between L3 1.0 to its value

            # Discount the rest proportionally
            for i in range(len(default_lambdas) - 1):
                modified[i] = (default_lambdas[i] / (sum(default_lambdas) - default_lambdas[3])) * (1 - modified[3])
                assert modified[i] >= 0
            assert math.isclose(sum(modified), 1, rel_tol=1e-9, abs_tol=0.0)

            # Compute cross entropy, save result ...
            cross_ent = compute_cross_entropy(model, modified, test_dataset, args.delim, start_op)
            print("\tCross-Entropy inc(L3,{}) = {:.4f}, H(p,q) = {:.4f}".format(inc, modified[3], cross_ent))
            save_line_to_csv(res_csv, "{},{},{},{},{}".format(lang, "inc", inc, modified[3], cross_ent), csv_vals)

        print("\t--------------------")
        for dec in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
            # Change L3
            modified[3] = L3 * dec  # set the trigram smoothing parameter to 90%, 80%, 70%, ... 10%, 0% of its value

            # Boost the rest proportionally
            for i in range(len(default_lambdas) - 1):
                modified[i] = (default_lambdas[i] / (sum(default_lambdas) - default_lambdas[3])) * (1 - modified[3])
                assert modified[i] >= 0
            assert math.isclose(sum(modified), 1, rel_tol=1e-9, abs_tol=0.0)

            # Compute cross entropy, save result ...
            cross_ent = compute_cross_entropy(model, modified, test_dataset, args.delim, start_op)
            print("\tCross-Entropy dec(L3,{}) = {:.4f}, H(p,q) = {:.4f}".format(dec, modified[3], cross_ent))
            save_line_to_csv(res_csv, "{},{},{},{},{}".format(lang, "dec", dec, modified[3], cross_ent), csv_vals)
        print("\t--------------------")
        print("\t--------------------")
