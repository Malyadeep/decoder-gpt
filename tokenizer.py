import torch
import torch.nn as nn
import math


class CharTokenizer:
    def __init__(self, text):
        self.vocab_text = text
        self.chars = sorted(set(self.vocab_text))

        # Character -> Integer ID
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

        # Integer ID -> Character
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        return "".join([self.itos[i] for i in tokens])


class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_length=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length

        pos_encoding = torch.zeros((max_length, embedding_dim))
        position = torch.arange(max_length)
        position = position.unsqueeze(1)
        temp_term = torch.exp(
            torch.arange(0, embedding_dim, 2) *
            math.log(10000) / embedding_dim
        )
        inv_term = 1 / temp_term
        pos_encoding[:, 0::2] = torch.sin(position * inv_term)
        pos_encoding[:, 1::2] = torch.cos(position * inv_term)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x):
        """
        Args
            x: input, (batch_size, sequence_length, embedding_dim)
        Returns
            positional encoded embeddings:
                (batch_size, sequence_length, embedding_dim)
        """
        input_size = x.size(1)
        if input_size > self.max_length:
            raise ValueError(
                f"Input sequence size {input_size} "
                f"exceeds maximum supported size {self.max_length}"
            )
        return x + self.pos_encoding[:input_size, :]

    def plot_pos_encoding(self):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        cmap = mpl.colormaps["jet"]
        plt.figure(1, figsize=(7, 6))
        plt.contourf(
            torch.arange(self.embedding_dim),
            torch.arange(self.max_length),
            self.pos_encoding,
            cmap=cmap
        )
        plt.colorbar()
        plt.show()
        plt.close()
