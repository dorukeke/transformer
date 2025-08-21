from os import path, makedirs
from re import split

import numpy as np
import torch

datafile = "./data/vocab.txt"


def encode_tokens_one_hot_encoding(tokens: [int], context_size: int = 1024, vocab_size: int = 1024) -> torch.Tensor:
    encoded_arr = np.zeros((max(context_size, len(tokens)), vocab_size))
    encoded_arr[np.arange(len(tokens)), tokens] = 1
    return torch.tensor(encoded_arr, dtype=torch.float32)


def decode_tokens_one_hot_encoding(tensor: torch.Tensor, context_size: int = 1024, vocab_size: int = 1024) -> [int]:
    max_indices = tensor.argmax(dim=1)
    return max_indices.take(max_indices.nonzero())


class Tokenizer:
    PAD_IDX = 0
    START_IDX = 1
    END_IDX = 2

    START = [START_IDX]
    END = [END_IDX]

    def __init__(self, vocab_size: int = 50000):
        super().__init__()
        self.tokens = [
            "[PAD]",
            "[SOS]",
            "[EOS]"
        ]
        self.vocab_size = vocab_size
        self.tokens_updated = False
        if path.exists(datafile):
            with open(datafile, "r") as f:
                self.tokens.extend([line.strip() for line in f.readlines()])

    def tokenize(self, text, with_sod: bool = True):
        text = self.normalise(text)
        tokenized_list = []
        if with_sod:
            tokenized_list.append(self.START_IDX)

        for word in Tokenizer.split(text):
            if word not in self.tokens:
                self.tokens_updated = True
                if len(self.tokens) >= self.vocab_size:
                    word = '[PAD]'
                else:
                    self.tokens.append(word)

            tokenized_list.append(self.tokens.index(word))

        if with_sod:
            tokenized_list.append(self.END_IDX)

        return tokenized_list

    @staticmethod
    def normalise(text):
        return text.lower()

    @staticmethod
    def split(text):
        return list(filter(None, split(r'\s+|([^0-9a-zA-ZıiİçÇşŞğĞüÜöÖ\s])', text)))

    def textual(self, tokenized_array):
        return " ".join([self.tokens[token_idx] for token_idx in tokenized_array if token_idx < len(self.tokens)])

    def decode(self, tokenized_array):
        return " ".join([self.tokens[token_idx] for token_idx in tokenized_array if 2 < token_idx < len(self.tokens)])

    def save(self):
        if not self.tokens_updated:
            return

        makedirs(path.dirname(datafile), exist_ok=True)
        with open(datafile, "w") as f:
            f.writelines([word + "\n" for word in self.tokens[3:]])


######################################################################################
#                                     TESTS
######################################################################################
def test_tokenizer():
    test_text = "Hello World!"
    test_token_len = 5
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(test_text)
    assert len(
        tokens) == test_token_len, f"Expected {test_token_len} token for: {test_text} but generated: {len(tokens)}."


def test_save_dictionary():
    test_text = "Hello World!"
    tokenizer = Tokenizer()
    tokenizer.tokenize(test_text)
    tokenizer.save()
    assert path.exists(datafile), f"Tokenizer.save() didn't generate expected file: {datafile}"


def test_functions():
    test_text = "Hello World!"
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(test_text)
    result = tokenizer.textual(tokens)
    assert result == "[SOS] hello world ! [EOS]", f"Textual representation: {result}, didn't match expected."

    result = tokenizer.decode(tokens)
    assert result == "hello world !", f"Textual representation: {result}, didn't match expected."


def test_one_hot_encoded_shape():
    test_text = "How are you today?"
    tokenizer = Tokenizer()
    out = encode_tokens_one_hot_encoding(tokenizer.tokenize(test_text))

    assert out.shape == torch.Size(
        [1024, 1024]
    ), f"Expected an output tensor of size (1024, 1024), got {out.shape}"


def test_one_hot_decoding():
    test_text = "How are you today?"
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(test_text)
    one_hot_tokens = encode_tokens_one_hot_encoding(tokens)
    out_tokens = decode_tokens_one_hot_encoding(one_hot_tokens)
    assert out_tokens == tokens, f"Expected {out_tokens} tokens from one hot decoding for {test_text}, got {tokens}"
