import torch
from tokenizer import CharTokenizer
from language_model import LanguageModel, generate_next_token


if __name__ == "__main__":
    # Create vocabulary and tokenizer
    vocab_text = "hello! how are you?"
    tokenizer = CharTokenizer(vocab_text)

    # Model configuration
    embedding_dim = 32
    num_layers = 1
    num_heads = 1

    # Language Model
    model = LanguageModel(
        tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )

    # Prompt
    text = "hello"
    print(f"Prompt: {text}")
    prompt_tokens = torch.tensor(tokenizer.encode(text)).unsqueeze(0)

    # Auto-regressive text generation
    max_output_length = 32
    generate_next_token(
        prompt_tokens,
        model,
        tokenizer,
        max_length=max_output_length
    )
