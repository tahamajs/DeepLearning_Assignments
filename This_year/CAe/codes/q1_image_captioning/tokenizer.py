"""
Tokenizer utilities for Image Captioning (build vocab, tokenization, detokenization)
"""
from collections import Counter
import re

class Tokenizer:
    def __init__(self, specials=['<pad>', '<start>', '<end>', '<unk>'], vocab_size=None):
        """Tokenizer with optional maximum vocabulary size.

        Args:
            specials: list of special tokens to include first
            vocab_size: optional int, maximum number of tokens INCLUDING specials
        """
        self.word2idx = {}
        self.idx2word = {}
        self.specials = specials
        # Backwards-compatible alias used in notebooks
        self.special_tokens = specials
        for i, tok in enumerate(specials):
            self.word2idx[tok] = i
            self.idx2word[i] = tok
        self.next_idx = len(specials)
        self.max_vocab = vocab_size  # None means unlimited

    def build_vocab(self, captions, min_freq=1):
        """Build vocab from list of captions (strings).

        If `self.max_vocab` is set, keep only the top-(max_vocab - n_specials)
        most frequent tokens (by corpus frequency) that meet `min_freq`.
        """
        cnt = Counter()
        for s in captions:
            toks = self._tokenize_text(s)
            cnt.update(toks)

        # Sort tokens by frequency (desc), then alphabetically for determinism
        items = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))

        # Determine cap if requested
        cap = None
        if self.max_vocab is not None:
            cap = max(0, int(self.max_vocab) - len(self.specials))

        added = 0
        for w, c in items:
            if c < min_freq:
                continue
            if w in self.word2idx:
                continue
            if cap is not None and added >= cap:
                break
            self.word2idx[w] = self.next_idx
            self.idx2word[self.next_idx] = w
            self.next_idx += 1
            added += 1

    def text_to_sequence(self, text):
        toks = self._tokenize_text(text)
        seq = [self.word2idx.get('<start>')]
        for t in toks:
            seq.append(self.word2idx.get(t, self.word2idx['<unk>']))
        seq.append(self.word2idx.get('<end>'))
        return seq

    def sequence_to_text(self, seq):
        words = [self.idx2word.get(i, '<unk>') for i in seq]
        # trim special tokens
        words = [w for w in words if w not in ['<start>', '<end>', '<pad>']]
        return ' '.join(words)

    def _tokenize_text(self, s):
        # Basic cleaning: remove punctuation except emojis and split on whitespace
        s = s.lower().strip()
        s = re.sub(r"[\"#$%&'()*+,-./:;<=>?@\[\\\]^_`{|}~]", '', s)
        toks = s.split()
        return toks

    def vocab_size(self):
        return len(self.word2idx)

    def __len__(self):
        return len(self.word2idx)
