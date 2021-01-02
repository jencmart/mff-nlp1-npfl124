def encode_as_utf(filename="dataset/TEXTCZ1.txt", endcoding="iso-8859-2"):
    with open(filename, "rt", encoding=endcoding) as f:
        lines = f.readlines()

    with open(filename + "-utf", "w") as f:
        for line in lines:
            f.write(line)


def load_dataset(file_path, frm=None, to=None, part=None, delim="/"):
    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    if frm is None:
        frm = 0
    if to is None:
        to = file_len(file_path)

    with open(file_path) as f:
        word_list = f.readlines()
        word_list = [w.strip() for w in word_list]

        if part is not None:
            word_list = [w.split(delim) for w in word_list]
            word_list = [w[part] for w in word_list]

        word_list = word_list[frm:to]
        lexicon = sorted(list(set(word_list)))
        charset = list(set([c for c in "".join(word_list)]))
        return word_list, lexicon, charset


class CountDict:
    def __init__(self):
        self._dict = {}

    def insert(self, key):
        if key in self._dict:
            self._dict[key] += 1
        else:
            self._dict[key] = 1

    def get_frequencies(self, denom, limit=None):
        for k, v in self._dict.items():
            if limit and v < limit:
                self._dict[k] = None
            else:
                self._dict[k] = v/denom
        return self._dict
