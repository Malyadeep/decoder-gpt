import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                "Embedding dimension not divisible by" +
                " number of attention heads"
            )
        self.head_dim = embedding_dim // num_heads

        # Create Query, Key and Value transformation
        self.Wq = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.Wk = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.Wv = nn.Linear(self.embedding_dim, self.embedding_dim)

        # Output map
        self.Wo = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, x):
        # Compute Query, Key and Value
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # Split Query, Key and Value across heads
        batch_size, N_tokens, embedding_dim = x.shape
        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Embedding dimensions does not match with Transformer design"
                f" Required: {self.embedding_dim}"
            )
        Q = Q.reshape(batch_size, N_tokens, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, N_tokens, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, N_tokens, self.num_heads, self.head_dim)

        # Make Query, Key and Value (B, H, N, head_dim) for tensor opeartions
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Make causal mask
        mask = torch.triu(
            torch.full(
                (N_tokens, N_tokens),
                fill_value=-float('inf'),
                device=x.device
            ),
            diagonal=1
        )

        # Compute attention scores
        attention_scores = Q @ K.transpose(-2, -1)
        attention_scores /= math.sqrt(self.head_dim)
        attention_scores = torch.softmax(attention_scores + mask, dim=-1)

        # Compute output from each head
        output = attention_scores @ V

        # Return Output to dimensions (B, N, H, head_dim) for concatenation
        output = output.transpose(1, 2)

        # Concatenate output from heads by reshaping
        # (H, head_dim) -> (H * head_dim) = (embedding_dim)
        output = output.reshape(batch_size, N_tokens, self.embedding_dim)

        # Output map after concatenation
        output = self.Wo(output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads=1):
        super().__init__()

        # Create attention head
        self.attention_head = MultiHeadAttention(
            embedding_dim,
            num_heads=num_heads
        )

        # Create layer norms
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        # Create shared MLP
        self.hidden_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, embedding_dim)
        )

    def forward(self, x):
        # Self attention block
        attention_output = self.attention_head(x)

        # Residual add + layer norm
        res_1 = self.layer_norm_1(attention_output + x)

        # Shared MLP
        mlp_out = self.mlp(res_1)

        # Residual add + layer norm
        output = self.layer_norm_2(mlp_out + res_1)

        return output
