import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.process_dataset import process_comment
from src.dataset import BookDataset
from src.model import BiLSTM


# ==========================================
# 1. Tests for Text Preprocessing (NLTK/Regex)
# ==========================================

def test_process_comment_removes_urls_and_ips() -> None:
    comment = "زر موقعنا https://example.com أو www.test.org على السيرفر 192.168.1.1"
    tokens = process_comment(comment)

    assert "رابط" in tokens
    assert "عنوان" in tokens
    assert "https://example.com" not in tokens
    assert "192.168.1.1" not in tokens


def test_process_comment_removes_emojis_punct_and_symbols() -> None:
    comment = "الكتاب رائع جداً!! 😍🔥، ولكن السعر 50$ غالي؟ «نعم»"
    tokens = process_comment(comment)

    for noise in ["😍", "🔥", "؟", "،", "$", "«", "»", "!"]:
        assert noise not in tokens

    assert "الكتاب" in tokens
    assert "رائع" in tokens


def test_process_comment_removes_arabic_stopwords() -> None:
    comment = "هذا الكتاب من أفضل الكتب التي قرأتها في حياتي"
    tokens = process_comment(comment)

    for stopword in ["هذا", "من", "التي", "في"]:
        assert stopword not in tokens

    assert "الكتاب" in tokens
    assert "أفضل" in tokens


def test_process_comment_empty_or_only_noise() -> None:
    assert process_comment("") == []
    assert process_comment("😂😂😂 !!! ؟؟؟") == []


# ==========================================
# 2. Tests for BookDataset
# ==========================================

@pytest.fixture
def sample_vocab() -> dict[str, int]:
    # Changed from "<PAD>" / "<UNK>" to "PAD" / "UNK" to match src/process_dataset.py
    return {"PAD": 0, "UNK": 1, "كتاب": 2, "ممتاز": 3, "سيء": 4}


def test_book_dataset_length_and_item_types(sample_vocab: dict[str, int]) -> None:
    tokens = [
        ["كتاب", "ممتاز"],
        ["كتاب", "سيء", "كلمة_غير_موجودة"],
    ]
    labels = np.array([1, 0])
    max_len = 5

    ds = BookDataset(tokens=tokens, labels=labels, vocab=sample_vocab, max_length=max_len)

    assert len(ds) == 2

    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert x.shape == (max_len,)
    assert y.item() == 1


def test_book_dataset_padding_and_truncation(sample_vocab: dict[str, int]) -> None:
    tokens = [
        ["كتاب"],                                      # Shorter -> pad with 0
        ["كتاب", "ممتاز", "سيء", "كتاب", "ممتاز"],     # Longer -> truncate to 3
    ]
    labels = np.array([1, 0])
    max_len = 3

    ds = BookDataset(tokens=tokens, labels=labels, vocab=sample_vocab, max_length=max_len)

    x_padded, _ = ds[0]
    x_truncated, _ = ds[1]

    assert x_padded.tolist() == [2, 0, 0]
    assert x_truncated.shape == (max_len,)
    assert x_truncated.tolist() == [2, 3, 4]


def test_book_dataset_unknown_token_mapping(sample_vocab: dict[str, int]) -> None:
    tokens = [["كتاب", "كلمة_مجهولة"]]
    labels = np.array([1])
    ds = BookDataset(tokens=tokens, labels=labels, vocab=sample_vocab, max_length=3)

    x, _ = ds[0]
    # "كتاب" -> 2, "كلمة_مجهولة" -> 1 (UNK), padding -> 0
    assert x.tolist() == [2, 1, 0]


def test_book_dataset_with_dataloader(sample_vocab: dict[str, int]) -> None:
    tokens = [["كتاب", "ممتاز"]] * 6
    labels = np.array([1, 0, 1, 0, 1, 0])
    ds = BookDataset(tokens=tokens, labels=labels, vocab=sample_vocab, max_length=4)

    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch_x, batch_y = next(iter(loader))

    assert batch_x.shape == (2, 4)
    assert batch_y.shape == (2,)


# ==========================================
# 3. Tests for BiLSTM Model
# ==========================================

def test_bilstm_forward_output_shape() -> None:
    vocab_size = 100
    embed_dim = 32
    hidden_size = 16
    num_classes = 2
    batch_size = 4
    seq_len = 10

    # Explicitly passed num_layers and dropout_rate
    model = BiLSTM(
        vocab_size=vocab_size,
        hidden_layer_size=hidden_size,
        input_size=embed_dim,
        output_size=num_classes,
        num_layers=1,
        dropout_rate=0.3,
    )

    dummy_input = torch.randint(1, vocab_size, (batch_size, seq_len))
    logits = model(dummy_input)

    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()


def test_bilstm_padding_idx_stays_zero() -> None:
    # Explicitly passed num_layers and dropout_rate
    model = BiLSTM(
        vocab_size=50,
        hidden_layer_size=16,
        input_size=8,
        output_size=2,
        num_layers=1,
        dropout_rate=0.3,
    )
    assert torch.all(model.embedding.weight[0] == 0)


def test_bilstm_backward_pass_updates_gradients() -> None:
    # Explicitly passed num_layers and dropout_rate
    model = BiLSTM(
        vocab_size=50,
        hidden_layer_size=16,
        input_size=8,
        output_size=2,
        num_layers=1,
        dropout_rate=0.3,
    )
    criterion = torch.nn.CrossEntropyLoss()

    x = torch.randint(1, 50, (4, 6))
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    assert model.linear.weight.grad is not None
    assert model.embedding.weight.grad is not None
