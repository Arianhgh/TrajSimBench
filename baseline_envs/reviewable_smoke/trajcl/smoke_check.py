import argparse
import sys

import torch

from config import Config


def main():
    parser = argparse.ArgumentParser(description="Tiny TrajCL smoke test")
    parser.parse_args()

    Config.update(
        {
            "device": torch.device("cpu"),
            "seq_embedding_dim": 8,
            "trans_hidden_dim": 16,
            "trans_attention_head": 2,
            "trans_attention_layer": 1,
            "trans_attention_dropout": 0.0,
            "trans_pos_encoder_dropout": 0.0,
            "moco_nqueue": 8,
            "moco_temperature": 0.05,
        }
    )

    from model.trajcl import TrajCL

    torch.manual_seed(7)
    model = TrajCL().to(Config.device)
    model.train()

    seq_len = 5
    batch = 2
    trajs1_emb = torch.randn(seq_len, batch, Config.seq_embedding_dim)
    trajs2_emb = torch.randn(seq_len, batch, Config.seq_embedding_dim)
    trajs1_spatial = torch.randn(seq_len, batch, 4)
    trajs2_spatial = torch.randn(seq_len, batch, 4)
    lens = torch.tensor([5, 4], dtype=torch.long)

    logits, targets = model(trajs1_emb, trajs1_spatial, lens, trajs2_emb, trajs2_spatial, lens)
    loss = model.loss(logits, targets)

    assert tuple(logits.shape) == (batch, 1 + Config.moco_nqueue)
    assert tuple(targets.shape) == (batch,)
    assert torch.isfinite(loss).item()

    print("PASS TrajCL smoke")
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"toy_loss={loss.item():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
