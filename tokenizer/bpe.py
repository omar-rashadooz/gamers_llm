"""Byte-level BPE tokenizer, written from scratch.

Idea: start from raw bytes (ids 0..255). Repeatedly find the most frequent
pair of neighbouring tokens and replace it with a NEW token. Each replacement
is called a "merge". After N merges we have a vocabulary of 256 + N tokens.
"""
import json
import re
from collections import Counter, defaultdict

EOT = "<|endoftext|>"  # special token: marks the end of one document

# Splits text into "words" (letters / digits / punctuation / whitespace).
# A leading space stays attached to the word, so " the" is one unit.
# Merges never cross these boundaries, which keeps tokens meaningful.
PAT = re.compile(r" ?[^\W\d_]+| ?\d+| ?_+| ?[^\s\w]+|\s+(?!\S)|\s+")


def merge_word(ids, pair, new_id):
    """Replace every occurrence of `pair` inside `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges=None):
        # merges: list of (a, b) pairs, in the order they were learned
        self.merges = [tuple(m) for m in (merges or [])]
        # pair -> id of the new token it creates (earlier merge = smaller id)
        self.ranks = {pair: 256 + i for i, pair in enumerate(self.merges)}
        # id -> the bytes this token stands for
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.ranks.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.eot_id = 256 + len(self.merges)
        self.vocab[self.eot_id] = EOT.encode("utf-8")
        self.vocab_size = self.eot_id + 1
        self._cache = {}

    # ------------------------------------------------------------ training
    @classmethod
    def train(cls, text, vocab_size=4096, verbose=True):
        n_merges = vocab_size - 256 - 1  # 256 bytes + 1 special token

        # 1) Count unique words once. Training on unique words (with their
        #    frequencies) is far faster than scanning the whole corpus.
        counts = Counter()
        for chunk in text.split(EOT):
            counts.update(PAT.findall(chunk))
        words = [list(w.encode("utf-8")) for w in counts]
        freqs = list(counts.values())
        if verbose:
            print(f"{len(words):,} unique words")

        # 2) Count every adjacent pair, and remember which words contain it.
        pair_counts = Counter()
        where = defaultdict(set)
        for i, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_counts[p] += freqs[i]
                where[p].add(i)

        # 3) Merge the most frequent pair, again and again.
        vocab = {i: bytes([i]) for i in range(256)}
        merges = []
        for k in range(n_merges):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)
            new_id = 256 + k
            merges.append(best)
            vocab[new_id] = vocab[best[0]] + vocab[best[1]]

            # only touch the words that actually contain `best`
            for i in list(where[best]):
                w, f = words[i], freqs[i]
                for p in zip(w, w[1:]):  # remove this word's old pair counts
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                w = merge_word(w, best, new_id)
                words[i] = w
                for p in zip(w, w[1:]):  # add its new pair counts
                    pair_counts[p] += f
                    where[p].add(i)
            where.pop(best, None)

            if verbose and (k < 10 or (k + 1) % 500 == 0):
                shown = vocab[new_id].decode("utf-8", errors="replace")
                print(f"merge {k + 1}/{n_merges}: {shown!r}")

        return cls(merges)

    # ------------------------------------------------------------ encoding
    def _encode_word(self, word):
        if word in self._cache:
            return self._cache[word]
        ids = list(word.encode("utf-8"))
        while len(ids) >= 2:
            # apply the earliest-learned merge that is possible
            pair = min(zip(ids, ids[1:]),
                       key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = merge_word(ids, pair, self.ranks[pair])
        self._cache[word] = ids
        return ids

    def encode(self, text):
        ids = []
        for j, part in enumerate(text.split(EOT)):
            if j > 0:
                ids.append(self.eot_id)
            for w in PAT.findall(part):
                ids.extend(self._encode_word(w))
        return ids

    def decode(self, ids):
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------- saving
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": self.merges}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["merges"])
