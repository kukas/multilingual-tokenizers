import math


# format function for the numbers
def humanize(n, base=1000):
    decimalnames = ["", "K", "M", "B"]
    binarynames = ["", "KiB", "MiB", "GiB"]
    names = binarynames if base == 1024 else decimalnames
    n = float(n)
    idx = max(
        0,
        min(
            len(names) - 1,
            int(math.floor(0 if n == 0 else math.log(abs(n)) / math.log(base))),
        ),
    )

    return "{:.1f}{}".format(n / base ** (idx), names[idx])


def read_vocab(vocab_file):
    with open(vocab_file) as tsv_file:
        first_column = []

        for line in tsv_file:
            first_column.append(line.rsplit("\t", 1)[0])
    return first_column


def load_lines(path, max_lines=None):
    lines = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            lines.append(line.rstrip())

    return lines


import numpy as np
import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize


def get_vocab_from_sp_model(sp):
    return [sp.id_to_piece(i) for i in range(len(sp))]


def compute_sorted_token_freqs(tokenized_text, vocabulary):
    # returns a list of tuples (freq, token)
    # and the frequency of the <unk> token

    freqs = [0] * len(vocabulary)
    for sentence in tokenized_text:
        for token in sentence:
            freqs[token] += 1
    assert vocabulary[0] == "<unk>"
    sorted_freqs = sorted(list(zip(freqs[1:], vocabulary[1:])), reverse=True)
    return sorted_freqs, freqs[0]


def compute_freqs(tokenized_text, vocab_size, return_probs=False):
    freqs = np.zeros(vocab_size)
    for sentence in tokenized_text:
        for token in sentence:
            freqs[token] += 1

    if return_probs:
        freqs = freqs / freqs.sum()

    return freqs


def compute_probs(freqs):
    return freqs / freqs.sum()


def compute_characters_per_token(tokenized_text, original_text, vocab):
    # caveat: adds up also unknown tokens
    total_tokens = 0
    total_chars = 0
    lengths = [0] * 32
    for token_sent, orig_sent in zip(tokenized_text, original_text):
        total_tokens += len(token_sent)
        total_chars += len(orig_sent)
        for token in token_sent:
            decoded = vocab[token]
            lengths[len(decoded)] += 1
        # for token in sentence:
        #     total_chars += len(token)
    return total_chars / total_tokens, lengths


def compute_characters_per_token_from_freqs(freqs, vocab, count_unk=False):
    total_tokens = 0
    total_chars = 0
    for token, freq in zip(vocab, freqs):
        if token == "<unk>":
            if count_unk:
                total_tokens += freq
                # total_chars += freq * 0  # count unk as 0 character (punishes cpt)
        else:
            total_tokens += freq
            total_chars += freq * len(token)
    return total_chars / total_tokens


def compute_average_rank(probs):
    probs = np.sort(probs)[::-1]
    ranks = np.arange(1, len(probs) + 1)
    return np.average(ranks, weights=probs)


def word_tokenize_list(lines, language="english", preserve_line=False):
    return [
        word_tokenize(line, language=language, preserve_line=False) for line in lines
    ]


def tokenize_words(words_text, tokenizer):
    tokens_per_word = []
    for sentence in words_text:
        tokens_per_word.append([])
        for i, word in enumerate(sentence):
            tokens_per_word[-1].append(tokenizer.encode(word))
    return tokens_per_word


# def compute_fertility(tokenized_text, original_text):
#     # subword fertility measures the average number of subwords produced per tokenized word.
#     # A minimum fertility of 1 means that the tokenizer’s vocabulary contains every single word in the text.
#     total_tokens = 0
#     total_words = 0
#     for token_sent, orig_sent in zip(tokenized_text, original_text):
#         words = word_tokenize(orig_sent, language, preserve_line=True)
#         total_words += len(words)
#         total_tokens += len(token_sent)
#     return total_tokens/total_words


def compute_fertility(words_text, tokenized_words_text):
    total_words = 0
    total_tokens = 0
    for words_sent, tokenized_words_sent in zip(words_text, tokenized_words_text):
        assert len(words_sent) == len(tokenized_words_sent)
        total_words += len(words_sent)
        total_tokens += sum([len(tokens) for tokens in tokenized_words_sent])

    return total_tokens / total_words


assert compute_fertility([["hello", "world"]], [[[1], [2]]]) == 1.0
assert compute_fertility([["hello", "world"]], [[[1, 2], [3, 4]]]) == 2.0


def compute_proportion_of_continued_words(words_text, tokenized_words_text):
    total_words = 0
    total_continued_words = 0
    for words_sent, tokenized_words_sent in zip(words_text, tokenized_words_text):
        assert len(words_sent) == len(tokenized_words_sent)
        total_words += len(words_sent)
        for word, tokens in zip(words_sent, tokenized_words_sent):
            if len(tokens) > 1:
                total_continued_words += 1
    return total_continued_words / total_words


assert compute_proportion_of_continued_words([["hello", "world"]], [[[1], [2]]]) == 0.0
assert (
    compute_proportion_of_continued_words([["hello", "world"]], [[[1, 2], [3, 4]]])
    == 1.0
)
assert (
    compute_proportion_of_continued_words([["hello", "world"]], [[[1, 2], [3]]]) == 0.5
)


def compute_average_log_probability(freqs, num_sentences):
    probs = compute_probs(freqs)
    # return np.sum(freqs * np.log(probs)) / np.sum(freqs)
    return np.sum(freqs * np.log(probs)) / num_sentences


def compute_entropy(probs):
    return -np.sum(probs * np.log(probs))


def compute_F_at_95(sorted_freqs):
    probs = compute_probs(sorted_freqs)
    cumsum = np.cumsum(probs)
    idx = np.where(cumsum > 0.95)[0][0]
    return sorted_freqs[idx]


def compute_divergence_from_uniform(freqs):
    p_uniform = 1 / len(freqs)
    probs = compute_probs(freqs)
    return np.sum(np.abs(probs - p_uniform)) / 2


from scipy.spatial.distance import jensenshannon


def compute_jsd(freqs1, freqs2):
    probs1 = compute_probs(freqs1)
    probs2 = compute_probs(freqs2)
    return jensenshannon(probs1, probs2)
