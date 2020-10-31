import math
import unittest

from cross_entropy import get_previous_words
from utils import load_dataset, file_len, compute_ngram
from utils import compute_counts
from utils import compute_conditional_probability

class TestLoading(unittest.TestCase):

    def test_linecount(self):
        l = file_len("../dataset/TEXTCZ1.txt-utf")
        self.assertEqual(l, 222412)
        l = file_len("../dataset/TEXTEN1.txt")
        self.assertEqual(l, 221098)

    def test_words_before(self):
        dataset = ["I", "am", "very", "hungry"]
        n = 3
        delim = " "
        start_op = "<S>"
        for idx in range(len(dataset)):
            words = get_previous_words(idx, n, dataset, delim, start_op)
            for w in words:
                print(w + " [" + dataset[idx] + "]")

    def print_ngram(self, ngram):
        val = list(ngram.values())[0]
        if isinstance(val, float):
            n = 1
        else:
            n = len(list(val.keys())[0].split(" "))+1
        print(">>>---  {}-gram  ----------------".format(n))

        entropy = 0
        for k, v in ngram.items():
            if isinstance(v, float):
                print("P({}) = {}".format(k, v))
                entropy += v*math.log2(v)
            else:
                # print("P({} | _)".format(k))
                for kk, vv in v.items():
                    print("P({} | {}) = {}".format(k, kk, vv))
                    entropy += vv * math.log2(vv)
                    # print(vv * math.log2(vv))
        if entropy != 0:
            entropy *= -1
        print("-----------------------------<<<")
        print("H({}-gram) = {}".format(n, entropy))
        print("-----------------------------<<<")

    def test_ngram(self):
        dataset = ["He", "can", "buy", "the", "can", "of", "soda", "."]
        uniform = compute_ngram(dataset, 0)
        # self.print_ngram(uniform)
        unigram = compute_ngram(dataset, 1)
        self.print_ngram(unigram)
        bigram = compute_ngram(dataset, 2)
        self.print_ngram(bigram)
        trigram = compute_ngram(dataset, 3)
        self.print_ngram(trigram)

    def test_cond_prob(self):
        dataset = ["a", "a", "a", "a", "a", "b", "b", "b", "c", "c"]  # a = 5 / ; b = 3 /  ; c = 2 /
        delim = " "

        n = 1
        wrd_probs = compute_counts(data=dataset, n=n, delim=delim, max_n=2)
        for k,v in wrd_probs.items():
            print("P({}) = {}".format(k, v))
        print("--------------------------")
        n = 2
        joint_probs = compute_counts(data=dataset, n=n, delim=delim, max_n=2)
        for k,v in joint_probs.items():
            print("P({}) = {}".format(k, v))
        print("--------------------------")

        cond_probs = compute_conditional_probability(w1_wm_probs=wrd_probs, w1_wm_wn_probs=joint_probs, delim=delim)
        prob_assert = {}
        for wn, dic in cond_probs.items():
            for w1_wn, p in dic.items():
                print("P({} | {}) = {}".format(wn, w1_wn, p))
                if w1_wn in prob_assert:
                    prob_assert[w1_wn] += p
                else:
                    prob_assert[w1_wn] = p
        print("--------------------------")
        for k, w in prob_assert.items():
            print("P( _ | {}) = {}".format(k, w))
            self.assertEqual(math.isclose(w, 1, rel_tol=1e-9, abs_tol=0.0), True)


    def test_counts(self):
        dataset = ["a", "a", "a", "a", "a", "b", "b", "b", "c", "c"]
        delim = " "

        n = 1
        count_prob = compute_counts(data=dataset, n=n, delim=delim)
        # self.assertEqual(count_prob, {'a': 0.5, 'b': 0.3, 'c': 0.2})
        i=0
        for w in count_prob.values():
            i+=w
        self.assertEqual(i, 1)
        n = 2
        count_prob = compute_counts(data=dataset, n=n, delim=delim)
        # self.assertEqual(count_prob, {'a a': 4/9, 'a b': 1/9, 'b b': 2/9,  'b c': 1/9, 'c c': 1/9})
        i=0
        for w in count_prob.values():
            i+=w
        self.assertEqual(i, 1)


    def test_load_dataset(self):
        d, lex, _ = load_dataset("../dataset/TEXTCZ1.txt-utf")
        self.assertEqual(len(d), 222412)
        self.assertEqual(len(lex), 42826)

        for word in d:
            self.assertEqual(word.strip(), word)
        d, lex, _ = load_dataset("../dataset/TEXTEN1.txt")
        self.assertEqual(len(d), 221098)
        self.assertEqual(len(lex), 9607)

        for word in d:
            self.assertEqual(word.strip(), word)

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()