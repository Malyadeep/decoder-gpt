import torch
import torch.nn as nn

from transformer import TransformerBlock
from tokenizer import PositionalEncoder


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=32,
        num_heads=1,
        num_layers=1
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Create Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim
        )

        # Create sinusoidal positional encoder
        self.pos_encoder = PositionalEncoder(
            embedding_dim=self.embedding_dim,
            max_length=1024
        )

        # Create and stack multiple transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(self.embedding_dim, self.num_heads)
             for _ in range(self.num_layers)]
        )

        # Final layer norm before vocabulary transformation
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        # Linear transform raw embeddings to vocabulary logits
        self.lm_transform = nn.Linear(
            self.embedding_dim,
            vocab_size
        )

    def forward(self, tokens):
        """
        Args:
            tokens: Input tokens (B, N)
        Returns:
            logits: Output logits (B, N, vocab_size)
        B: Batch size
        N: No. of tokens
        vocab_size: total size of vocabulary
        """
        # Create token embeddings
        embeddings = self.embedding(tokens)

        # Include positional encoding
        input = self.pos_encoder(embeddings)

        # Pass the input through transformer layers
        for _, layer in enumerate(self.layers):
            input = layer(input)

        # Final layer norm before transformation to vocabulary
        normed_output = self.final_layer_norm(input)

        # Transform raw embeddings to vocabulary logits
        logits = self.lm_transform(normed_output)

        return logits


@torch.no_grad
def generate_next_token(
    prompt_tokens,
    model,
    tokenizer,
    max_length=32
):
    model.eval()
    tokens = prompt_tokens.clone()
    for _ in range(max_length):
        # Evaluate the model to get logits over vocabulary
        logits = model(tokens)

        # Get probability distribution over vocabulary for last logit
        probability = torch.softmax(logits[:, -1, :], dim=-1)

        # Sample the next token from the probability distribution
        output_token = torch.argmax(probability, dim=1, keepdim=True)
        tokens = torch.cat([tokens, output_token], dim=1)
    token_string = tokenizer.decode(tokens[0].tolist())
    print(f"Generated output: {token_string}")
