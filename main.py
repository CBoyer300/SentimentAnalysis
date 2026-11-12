"""
Preprocessing for a sentiment-analysis transformer.

Pipeline:
    parse review files -> tokenize -> split train/val/test
    -> build vocab (train only) -> encode + pad
"""

import os
import re
import random
import xml.etree.ElementTree as ET
from collections import Counter

# ---------------------------------------------------------------------------
# Hyperparameters / config
# ---------------------------------------------------------------------------
DATA_DIR = "sorted_data"
STOPWORDS_PATH = os.path.join(DATA_DIR, "stopwords")

MIN_WORD_COUNT = 5        # keep words that appear at least this many times
MAX_LEN = 512             # final sequence length, including [BOS] and [EOS]
REMOVE_STOPWORDS = False  # negations/"but" are always kept even if True
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1            # remainder becomes the test set
SEED = 42

PAD, UNK, BOS, EOS = "[PAD]", "[UNK]", "[BOS]", "[EOS]"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]  # IDs 0, 1, 2, 3
MAX_TOKENS = MAX_LEN - 2               # room left for BOS and EOS

# Words that carry sentiment and should survive stopword removal
KEEP_WORDS = {
    "no", "nor", "not", "never", "none", "nothing", "neither", "cannot",
    "but", "however", "very", "too", "only", "against",
}

# ---------------------------------------------------------------------------
# Text cleaning / tokenization
# ---------------------------------------------------------------------------
_AMP_RE = re.compile(
    r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)"
)
_INVALID_XML_CHARS_RE = re.compile(
    "[^\t\n\r\u0020-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]"
)
_TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]")


def clean_xml(text):
    """Escape bare ampersands and strip characters that are illegal in XML 1.0."""
    text = _AMP_RE.sub("&amp;", text)
    return _INVALID_XML_CHARS_RE.sub("", text)


def tokenize(text):
    """Lowercase and split into word tokens, with punctuation as separate tokens."""
    return _TOKEN_RE.findall(text.lower())


def load_stopwords(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return set(f.read().lower().split())


def remove_stopwords(tokens, stop_words):
    return [
        t for t in tokens
        if t not in stop_words or t in KEEP_WORDS or t.endswith("n't")
    ]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def get_review_files(directory):
    """Return a list of (file_path, label) for every domain folder. 1 = positive, 0 = negative."""
    folders = sorted(entry.path for entry in os.scandir(directory) if entry.is_dir())

    files = []
    for folder in folders:
        for name, label in (("positive.review", 1), ("negative.review", 0)):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                files.append((path, label))
    return files


def get_text(elem, tag):
    """Text of a child tag, or '' if the tag or its text is missing."""
    child = elem.find(tag)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def parse_file(path, label, stop_words):
    """Parse one .review file into a list of review dicts."""
    print(path)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    # The files have no single root element, so wrap them in one
    root = ET.fromstring("<reviews>" + clean_xml(raw) + "</reviews>")

    reviews = []
    for elem in root.findall("review"):
        text = get_text(elem, "review_text")
        if not text:
            continue

        tokens = tokenize(text)
        if REMOVE_STOPWORDS:
            tokens = remove_stopwords(tokens, stop_words)
        tokens = tokens[:MAX_TOKENS]  # truncate instead of dropping long reviews

        rating_str = get_text(elem, "rating")
        reviews.append({
            "rating": float(rating_str) if rating_str else None,
            "title": get_text(elem, "title"),
            "tokens": tokens,
            "label": label,
        })
    return reviews


def load_dataset(directory, stop_words):
    dataset = []
    for path, label in get_review_files(directory):
        dataset.extend(parse_file(path, label, stop_words))
    return dataset


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_dataset(dataset):
    dataset = list(dataset)
    random.Random(SEED).shuffle(dataset)

    n = len(dataset)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    return (
        dataset[:n_train],
        dataset[n_train:n_train + n_val],
        dataset[n_train + n_val:],
    )


# ---------------------------------------------------------------------------
# Vocabulary / encoding
# ---------------------------------------------------------------------------
def build_vocab(train_set):
    """Special tokens first, then every training word seen >= MIN_WORD_COUNT times."""
    counts = Counter()
    for review in train_set:
        counts.update(review["tokens"])

    words = [w for w, c in counts.most_common() if c >= MIN_WORD_COUNT]
    itos = SPECIAL_TOKENS + words
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def encode(tokens, stoi):
    """List of word tokens -> list of IDs (unknown words become [UNK])."""
    unk = stoi[UNK]
    return [stoi.get(t, unk) for t in tokens]


def decode(ids, itos):
    """List of IDs -> string, dropping padding."""
    return " ".join(itos[i] for i in ids if itos[i] != PAD)


def encode_review(tokens, stoi):
    """Add [BOS]/[EOS], pad to MAX_LEN, and return (input_ids, attention_mask)."""
    ids = [stoi[BOS]] + encode(tokens, stoi) + [stoi[EOS]]
    n_pad = MAX_LEN - len(ids)

    attention_mask = [1] * len(ids) + [0] * n_pad
    ids = ids + [stoi[PAD]] * n_pad
    return ids, attention_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    stop_words = load_stopwords(STOPWORDS_PATH)

    dataset = load_dataset(DATA_DIR, stop_words)
    train, val, test = split_dataset(dataset)

    stoi, itos = build_vocab(train)  # train only, so val/test words don't leak in

    for split in (train, val, test):
        for review in split:
            review["input_ids"], review["attention_mask"] = encode_review(
                review["tokens"], stoi
            )

    print(f"Reviews: {len(dataset)} "
          f"(train {len(train)}, val {len(val)}, test {len(test)})")
    print(f"Vocab size: {len(stoi)}")
    sample = train[0]
    print(f"Sample label: {sample['label']}, rating: {sample['rating']}")
    print(decode(sample["input_ids"], itos))

    return train, val, test, stoi, itos


if __name__ == "__main__":
    main()