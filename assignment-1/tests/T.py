import math
import unittest

from perform_experiments import load_dataset
from perform_experiments import compute_counts
from perform_experiments import compute_conditional_probability

class TestLoading(unittest.TestCase):

    def test_cond_prob(self):
        dataset = ["a", "a", "a", "a", "a", "b", "b", "b", "c", "c"]
        delim = " "

        n = 1
        wrd_probs = compute_counts(dataset_list=dataset, n=n, delim=delim)

        n = 2
        joint_probs = compute_counts(dataset_list=dataset, n=n, delim=delim)

        cond_probs = compute_conditional_probability(w1_wm_probs=wrd_probs, joint_probs=joint_probs, delim=delim)
        prob_assert = {}
        for wn, dic in cond_probs.items():
            for w1_wn, p in dic.items():
                print("P({} | {}) = {}".format(wn, w1_wn, p))
                if w1_wn in prob_assert:
                    prob_assert[w1_wn] += p
                else:
                    prob_assert[w1_wn] = p
        for k, w in prob_assert.items():
            self.assertEqual(math.isclose(w, 1, rel_tol=1e-9, abs_tol=0.0), True)


    def test_counts(self):
        dataset = ["a", "a", "a", "a", "a", "b", "b", "b", "c", "c"]
        delim = " "

        n = 1
        count_prob = compute_counts(dataset_list=dataset, n=n, delim=delim)
        self.assertEqual(count_prob, {'a': 0.5, 'b': 0.3, 'c': 0.2})

        n = 2
        count_prob = compute_counts(dataset_list=dataset, n=n, delim=delim)
        self.assertEqual(count_prob, {'a a': 4/9, 'a b': 1/9, 'b b': 2/9,  'b c': 1/9, 'c c': 1/9})

        print(count_prob)

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