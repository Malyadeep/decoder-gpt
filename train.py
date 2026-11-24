import torch

from tokenizer import CharTokenizer
from language_model import LanguageModel


class TrainLanguageModel:
    def __init__(
        self,
        corpus_file
    ):
        self.corpus_file = corpus_file
        with open(self.corpus_file, "r") as f:
            self.corpus_text = f.read()
        self.tokenizer = CharTokenizer(self.corpus_text)

    def create_datasets(
        self,
        training_batch_size=32,
        validation_batch_size=32,
        block_size=16
    ):
        self.block_size = block_size
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        train_split = 0.8
        idx = int(train_split*len(self.corpus_text))
        training_text = self.corpus_text[:idx]
        validation_text = self.corpus_text[idx:]
        training_tokens = torch.tensor(
            self.tokenizer.encode(training_text),
            dtype=torch.long
        )
        validation_tokens = torch.tensor(
            self.tokenizer.encode(validation_text),
            dtype=torch.long
        )
        self.training_batch_x, self.training_batch_y = get_batch(
            training_tokens,
            self.training_batch_size,
            self.block_size
        )
        self.validation_batch_x, self.validation_batch_y = get_batch(
            validation_tokens,
            self.validation_batch_size,
            self.block_size
        )

    def setup_model(
        self,
        embedding_dim=32,
        num_heads=1,
        num_layers=1
    ):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.model = LanguageModel(
            self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        )

    def count_model_parameters(self):
        if not hasattr(self, "model"):
            raise ValueError("model is not yet set up in trainer")
        total_parameters = sum(
            p.numel() for p in self.model.parameters()
        )
        trainable_parameters = sum(
            p.numel() for p in self.model.parameters()
            if p.requires_grad
        )
        print(f"{'Total no. of parameters':50}: {total_parameters}")
        print(f"{'Total no. of trainable parameters':50}: "
              f"{trainable_parameters}")
        print("\nTotal parameter audit\n")
        print(f"{'name':<50} | {'requires_grad':<15} | {'no.of params':<30} ")
        for name, param in self.model.named_parameters():
            print(f"{name:<50} | {param.requires_grad:<15} | "
                  f"{param.numel():<30}")


def get_batch(tokens, batch_size, block_size):
    # Randomly sample starting position
    starting_pos = torch.randint(
        0,
        len(tokens) - block_size,
        (batch_size,)
    )

    # Construct input pair from the starting position
    x = torch.stack(
        [tokens[pos: pos + block_size]
         for pos in starting_pos]
    )

    # Construct output pair by shifting the token indices by 1
    y = torch.stack(
        [tokens[pos + 1: pos+block_size + 1]
         for pos in starting_pos]
    )

    return x, y


if __name__ == "__main__":
    # Create trainer with text corpus
    corpus_file = "corpus.txt"
    trainer = TrainLanguageModel(corpus_file)

    # Create datasets with specified batch and block sizes
    training_batch_size = 32
    validation_batch_size = 32
    block_size = 16
    trainer.create_datasets(
        training_batch_size=training_batch_size,
        validation_batch_size=validation_batch_size,
        block_size=block_size
    )

    # Setup language model with specified model configuration
    embedding_dim = 32
    num_layers = 1
    num_heads = 1
    trainer.setup_model(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )
    trainer.count_model_parameters()
