import os
import re
import random
import xml.etree.ElementTree as ET
from collections import Counter
import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.nn import functional as F # type: ignore

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using: ", device)

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
SEED = 1337

torch.manual_seed(SEED)

#Transformer hyperparameters
# Hyperparameters:
batch_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
eval_iters = 200
n_embd = 64
n_head = 2
n_layers = 2
dropout = 0.2

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
# Transformer + related classes
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(), #Try GELU after testint ReLU
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_mask):
        # x: (B, T, C), key_mask: (B, T) bool, True = real token
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)
        wei = wei.masked_fill(~key_mask[:, None, :], float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        return wei @ self.value(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList(Head(n_embd, head_size) for _ in range(num_heads))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_mask):
        out = torch.cat([h(x, key_mask) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadedAttention(n_embd, n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, key_mask):
        x = x + self.sa(self.ln1(x), key_mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class ReviewTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, padding_idx=0)
        self.position_embedding_table = nn.Embedding(MAX_LEN, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(n_embd, n_head) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(n_embd)
        self.classifier = nn.Linear(n_embd, 1)

    def forward(self, idx, attention_mask, labels=None):
        B, T = idx.shape
        key_mask = attention_mask.bool()

        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, key_mask)
        x = self.ln_f(x)

        # masked mean pooling over real tokens only
        m = attention_mask.unsqueeze(-1).to(x.dtype)           # (B, T, 1)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

        logits = self.classifier(pooled).squeeze(-1)           # (B,)
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        return logits, loss
    
def to_tensors(split):
    ids = torch.tensor([r["input_ids"] for r in split])
    mask = torch.tensor([r["attention_mask"] for r in split])
    labels = torch.tensor([r["label"] for r in split], dtype=torch.float)
    return ids, mask, labels

def get_batch(data):
    ids, mask, labels = data
    ix = torch.randint(len(ids), (batch_size,))
    xb = ids[ix].to(device)
    yb = mask[ix].to(device)
    zb = labels[ix].to(device)
    return xb, yb, zb

@torch.no_grad()
def evaluate(model, data, bs=128):
    #Set to eval mode
    model.eval()
    ids, mask, labels = data
    n = len(ids)
    total_loss, correct = 0.0, 0
    for i in range(0, n, bs):
        x = ids[i:i + bs].to(device)
        m = mask[i:i + bs].to(device)
        y = labels[i:i + bs].to(device)
        L = int(m.sum(1).max())
        logits, loss = model(x[:, :L], m[:, :L], y)
        total_loss += loss.item() * len(y)
        correct += ((logits > 0).float() == y).sum().item()

    #Set back to training mode
    model.train()
    return total_loss / n, correct / n 


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():

    stop_words = load_stopwords(STOPWORDS_PATH)

    dataset = load_dataset(DATA_DIR, stop_words)
    train, val, test = split_dataset(dataset)

    stoi, itos = build_vocab(train)  # train only, so val/test words don't leak in
    vocab_size = len(stoi)

    t = ReviewTransformer(vocab_size).to(device)
    print(f"{sum(p.numel() for p in t.parameters())} parameters")

    optimizer = torch.optim.AdamW(t.parameters(), lr=learning_rate)

    for split in (train, val, test):
        for review in split:
            review["input_ids"], review["attention_mask"] = encode_review(
                review["tokens"], stoi
            )

    train_data = to_tensors(train)
    val_data = to_tensors(val)

    print(f"Reviews: {len(dataset)} " f"(train {len(train)}, val {len(val)}, test {len(test)})")
    print(f"Vocab size: {vocab_size}")
    sample = train[0]
    print(f"Sample label: {sample['label']}, rating: {sample['rating']}")
    print(decode(sample["input_ids"], itos))

    #Sample batch -> logits/loss
    xb, yb, zb = get_batch(train_data)
    logits, loss = t(xb, yb, zb)
    print(logits)
    print(logits.size())
    print(loss)

    best_val = float('inf')
    running = 0.0

    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            val_loss, val_acc = evaluate(t, val_data)
            print(f"step {step}: train {running / eval_interval:.4f} | " f"val {val_loss:.4f} | val acc {val_acc:.3f}")
            running = 0.0
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": t.state_dict(), "stoi": stoi, "itos": itos}, "best.pt")

        xb, yb, zb = get_batch(train_data)
        logits, loss = t(xb, yb, zb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        

    return train, val, test, stoi, itos


if __name__ == "__main__":
    main()