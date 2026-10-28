from transformer.data import EnglishDataset, SimpleTokenizer, make_lm_example


def test_分词器编码解码():
    tokenizer = SimpleTokenizer()
    ids = tokenizer.encode("The cat is big.")
    assert ids[0] == tokenizer.vocab_to_id["<BOS>"]
    assert ids[-1] == tokenizer.vocab_to_id["<EOS>"]
    assert tokenizer.decode(ids) == "the cat is big"


def test_不认识的词变成UNK():
    tokenizer = SimpleTokenizer()
    assert tokenizer.encode("zebra")[1] == tokenizer.vocab_to_id["<UNK>"]


def test_目标是输入右移一位():
    x, y = make_lm_example([5, 6, 7, 8], seq_length=3, pad_id=0)
    assert x == [5, 6, 7]
    assert y == [6, 7, 8]


def test_padding位置的目标是负1():
    x, y = make_lm_example([5, 6], seq_length=4, pad_id=0)
    assert x == [5, 6, 0, 0]
    assert y == [6, -1, -1, -1]


def test_数据集样本():
    tokenizer = SimpleTokenizer()
    dataset = EnglishDataset(tokenizer, seq_length=12, num_samples=5)
    assert len(dataset) == 5
    sample = dataset[0]
    assert sample["x"].shape == (12,) and sample["y"].shape == (12,)
    assert sample["x"][0].item() == tokenizer.vocab_to_id["<BOS>"]


def test_太长的句子被丢掉():
    tokenizer = SimpleTokenizer()
    dataset = EnglishDataset(tokenizer, seq_length=4, sentences=["the cat", "the cat is big and old"])
    assert len(dataset) == 1
