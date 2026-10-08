"""TinyStories -> 8,192-token byte-level BPE -> flat uint16 token files.

Only the first ~250k stories of the train split are streamed (52M tokens, enough for the 50M the runs use);
nothing else is downloaded. Stories are separated by an <|eos|> token.
"""
import os
import json
import numpy as np

VOCAB_SIZE = 8192
EOS = "<|eos|>"


def _stream(split):
    from datasets import load_dataset
    for row in load_dataset("roneneldan/TinyStories", split=split, streaming=True):
        text = row["text"].strip()
        if text:
            yield text


def train_tokenizer(out_dir, n_stories=100_000):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=[EOS],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    def it():
        for i, t in enumerate(_stream("train")):
            if i >= n_stories:
                return
            yield t

    tok.train_from_iterator(it(), trainer=trainer)
    path = os.path.join(out_dir, "tokenizer.json")
    tok.save(path)
    return path


def load_tokenizer(out_dir):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(out_dir, "tokenizer.json"))


def tokenize_split(tok, split, n_tokens, path, batch=2000):
    """Write the first n_tokens tokens of a split to a uint16 file."""
    eos = tok.token_to_id(EOS)
    out = np.empty(n_tokens, dtype=np.uint16)
    filled, n_docs, buf = 0, 0, []

    def flush(buf):
        nonlocal filled, n_docs
        for enc in tok.encode_batch(buf):
            ids = enc.ids + [eos]
            take = min(len(ids), n_tokens - filled)
            out[filled:filled + take] = ids[:take]
            filled += take
            n_docs += 1
            if filled >= n_tokens:
                return True
        return False

    for text in _stream(split):
        buf.append(text)
        if len(buf) == batch:
            if flush(buf):
                break
            buf = []
    else:
        flush(buf)
    out[:filled].tofile(path)
    return filled, n_docs


def prepare(out_dir="data", n_train=52_000_000, n_val=1_000_000):
    """Idempotent: skips any file that already exists. 52M train tokens gives 50M plus slack."""
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    if not os.path.exists(os.path.join(out_dir, "tokenizer.json")):
        train_tokenizer(out_dir)
    tok = load_tokenizer(out_dir)
    n_tr, d_tr = tokenize_split(tok, "train", n_train, os.path.join(out_dir, "train.bin"))
    n_va, d_va = tokenize_split(tok, "validation", n_val, os.path.join(out_dir, "val.bin"))
    meta = {"vocab_size": tok.get_vocab_size(), "eos_id": tok.token_to_id(EOS),
            "train_tokens": n_tr, "train_stories": d_tr, "val_tokens": n_va, "val_stories": d_va}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    return meta


class TokenLoader:
    """Random windows of seq_len+1 tokens from a flat token file, sampled without replacement
    so that one pass of n_steps * batch * seq_len tokens reads each window once."""

    def __init__(self, path, batch, seq_len, device, seed=0):
        self.data = np.fromfile(path, dtype=np.uint16)
        self.batch, self.seq_len, self.device = batch, seq_len, device
        self.n_windows = (len(self.data) - 1) // seq_len
        self.rng = np.random.default_rng(seed)
        self._order, self._pos = self.rng.permutation(self.n_windows), 0

    def next(self):
        import torch
        if self._pos + self.batch > self.n_windows:
            self._order, self._pos = self.rng.permutation(self.n_windows), 0
        idx = self._order[self._pos:self._pos + self.batch] * self.seq_len
        self._pos += self.batch
        x = np.stack([self.data[i:i + self.seq_len + 1] for i in idx]).astype(np.int64)
        x = torch.from_numpy(x).pin_memory().to(self.device, non_blocking=True)
        return x[:, :-1], x[:, 1:]

    def fixed_batches(self, n, batch):
        """Deterministic evaluation batches from the start of the file."""
        import torch
        out = []
        for b in range(n):
            idx = (np.arange(batch) + b * batch) * self.seq_len
            x = np.stack([self.data[i:i + self.seq_len + 1] for i in idx]).astype(np.int64)
            x = torch.from_numpy(x).to(self.device)
            out.append((x[:, :-1], x[:, 1:]))
        return out
