"""
语言模型的数据：一个词表很小的分词器和一组简单的英文句子。
"""
import torch
from torch.utils.data import Dataset

SENTENCES = [
    "the cat is sleeping on the bed",
    "i like to eat good food",
    "the dog runs in the park",
    "she reads a book every day",
    "we go to school in the morning",
    "the bird sings in the tree",
    "he drives a red car",
    "they play with their friends",
    "the house is big and beautiful",
    "water is cold and fresh",
    "the boy likes to play games",
    "she has a small cat",
    "we see the blue sky",
    "the man works in the office",
    "children like to eat ice cream",
    "the woman cooks good food",
    "birds fly in the sky",
    "fish swim in the water",
    "the car is new and fast",
    "people walk in the street",
    "the girl reads a good book",
    "cats like to sleep here",
    "dogs run and play outside",
    "the family lives in a house",
    "friends come to visit today",
    "the tree is green and tall",
    "i want to go home now",
    "you can see the red bird",
    "we will eat dinner soon",
    "they should go to bed",
]


class SimpleTokenizer:
    def __init__(self):
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'

        basic_vocab = [
            self.pad_token, self.unk_token, self.bos_token, self.eos_token,
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those',
            'is', 'am', 'are', 'was', 'were', 'be', 'being', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should', 'may', 'might',
            'cat', 'dog', 'bird', 'fish', 'house', 'car', 'tree', 'book', 'water', 'food',
            'man', 'woman', 'child', 'boy', 'girl', 'people', 'person', 'family', 'friend',
            'good', 'bad', 'big', 'small', 'hot', 'cold', 'new', 'old', 'red', 'blue', 'green',
            'go', 'come', 'see', 'look', 'hear', 'say', 'tell', 'know', 'think', 'want', 'like',
            'eat', 'drink', 'sleep', 'work', 'play', 'run', 'walk', 'sit', 'stand', 'live',
            'today', 'yesterday', 'tomorrow', 'now', 'here', 'there', 'yes', 'no', 'please', 'thank'
        ]
        self.vocab_to_id = {word: idx for idx, word in enumerate(basic_vocab)}
        self.id_to_vocab = {idx: word for idx, word in enumerate(basic_vocab)}
        self.vocab_size = len(basic_vocab)
        self.pad_id = self.vocab_to_id[self.pad_token]

    def encode(self, text):
        """文本 -> [BOS, token, ..., EOS]，不在词表里的词变成 UNK"""
        tokens = [self.bos_token]
        for word in text.lower().split():
            word = word.strip('.,!?;:"()[]')  # 简单处理标点
            tokens.append(word if word in self.vocab_to_id else self.unk_token)
        tokens.append(self.eos_token)
        return [self.vocab_to_id[token] for token in tokens]

    def decode(self, token_ids):
        """去掉 PAD、BOS、EOS 后拼回文本"""
        special = {self.pad_token, self.bos_token, self.eos_token}
        words = [self.id_to_vocab[token_id] for token_id in token_ids if token_id in self.id_to_vocab]
        return ' '.join(word for word in words if word not in special)


def make_lm_example(token_ids, seq_length, pad_id):
    """
    语言模型的样本: 输入是前 seq_length 个 token，目标是右移一位后的序列。
    不足的部分补 pad，目标里 pad 的位置标成 -1，算损失时跳过。
    """
    padded = (token_ids + [pad_id] * (seq_length + 1))[:seq_length + 1]
    x = padded[:-1]
    y = [-1 if token == pad_id else token for token in padded[1:]]
    return x, y


class EnglishDataset(Dataset):
    def __init__(self, tokenizer, seq_length, num_samples=None, sentences=SENTENCES):
        if num_samples:
            sentences = (sentences * (num_samples // len(sentences) + 1))[:num_samples]

        self.examples = []
        for sentence in sentences:
            encoded = tokenizer.encode(sentence)
            if len(encoded) <= seq_length + 1:  # 太长的句子丢掉
                self.examples.append(make_lm_example(encoded, seq_length, tokenizer.pad_id))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        return {
            'x': torch.tensor(x, dtype=torch.long),
            'y': torch.tensor(y, dtype=torch.long)
        }
