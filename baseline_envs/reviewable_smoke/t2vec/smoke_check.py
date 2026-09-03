import argparse
import sys

import numpy as np
import torch

if not hasattr(np, "int"):
    np.int = int

from models import EncoderDecoder
from train import NLLcriterion, genLoss
from collections import namedtuple


def main():
    parser = argparse.ArgumentParser(description="Tiny t2vec smoke test")
    parser.parse_args()

    torch.manual_seed(7)

    TD = namedtuple("TD", ["src", "lengths", "trg", "invp"])
    batch = TD(
        src=torch.tensor([[4, 8], [5, 9], [6, 10], [7, 0]], dtype=torch.long),
        lengths=torch.tensor([[4, 3]], dtype=torch.long),
        trg=torch.tensor([[2, 2], [5, 9], [6, 10], [7, 3], [3, 0]], dtype=torch.long),
        invp=[],
    )

    class Args:
        cuda = False
        generator_batch = 8

    vocab_size = 32
    model = EncoderDecoder(
        vocab_size=vocab_size,
        embedding_size=8,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        bidirectional=True,
    )
    generator = torch.nn.Sequential(torch.nn.Linear(8, vocab_size), torch.nn.LogSoftmax(dim=1))

    output = model(batch.src, batch.lengths, batch.trg)
    loss = genLoss(batch, model, generator, NLLcriterion(vocab_size), Args())

    assert tuple(output.shape) == (batch.trg.shape[0] - 1, batch.trg.shape[1], 8)
    assert torch.isfinite(loss).item()

    print("PASS t2vec smoke")
    print(f"forward_output_shape={tuple(output.shape)}")
    print(f"toy_loss={loss.item():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
