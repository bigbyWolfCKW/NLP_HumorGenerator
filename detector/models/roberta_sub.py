import re

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import AutoModel


class ContextSUBModel(nn.Module):
    def __init__(self, backbone, use_sub):
        super().__init__()
        self.use_sub = use_sub
        self.backbone = AutoModel.from_pretrained(backbone, local_files_only=True, add_pooling_layer=False)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(self.backbone.config.hidden_dropout_prob)
        self.dense = nn.Linear(hidden + 64, hidden)
        self.output = nn.Linear(hidden, 2)

    def forward(self, ids, mask, features):
        context = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
        if not self.use_sub:
            features = torch.zeros_like(features)
        joined = torch.cat([context, features], dim=1)
        hidden = torch.tanh(self.dense(self.dropout(joined)))
        return self.output(self.dropout(hidden))


class SubjectivityLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(.3)
        self.classifier = nn.Linear(hidden_dim, 1)

    def encode(self, ids, lengths):
        packed = pack_padded_sequence(self.embedding(ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return hidden[-1]


def sub_inputs(texts, vocab, config):
    maximum = config["max_length"]
    if config.get("pad_id", 0) != 0 or config.get("unk_id", 1) != 1:
        raise ValueError("Unsupported SUB padding/unknown token IDs.")
    sequences, lengths = [], []
    for text in texts:
        if config.get("lowercase", True):
            text = text.lower()
        tokens = re.findall(config.get("tokenizer_pattern", r"\w+|[^\w\s]"), text)[:maximum]
        ids = [vocab.get(token, 1) for token in tokens] or [1]
        lengths.append(len(ids))
        sequences.append(ids + [0] * (maximum - len(ids)))
    return torch.tensor(sequences, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)
