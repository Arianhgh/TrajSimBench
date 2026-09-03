import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "traj_rnns"))
sys.path.insert(0, ROOT)

from traj_model import make_model, subsequent_mask


def main():
    parser = argparse.ArgumentParser(description="Tiny TMN smoke test")
    parser.parse_args()

    torch.manual_seed(7)

    model = make_model(src_vocab=64, tgt_vocab=64, N=1, d_model=8, d_ff=16, h=2, dropout=0.0)
    src = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    src_mask = (src != 0).unsqueeze(-2)
    src_lengths = [3, 2]

    encoded = model(src, src_mask, src_lengths)
    tgt = torch.tensor([[1, 2, 3], [1, 4, 0]], dtype=torch.long)
    tgt_mask = (tgt != 0).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
    decoded = model.decode(model.encode(src, src_mask), src_mask, tgt, tgt_mask)

    assert tuple(encoded.shape) == (2, 8)
    assert tuple(decoded.shape) == (2, 3, 8)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = encoded.square().mean() + decoded.square().mean()
    loss.backward()
    grad_norms = [
        parameter.grad.detach().norm()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert grad_norms and all(torch.isfinite(norm) for norm in grad_norms)
    before_step = next(model.parameters()).detach().clone()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before_step, next(model.parameters()).detach())

    scores = encoded.detach() @ encoded.detach().T
    ranking = scores.argsort(dim=1, descending=True)
    assert tuple(ranking.shape) == (2, 2)

    print("PASS TMN smoke")
    print(f"encoder_output_shape={tuple(encoded.shape)}")
    print(f"decoder_output_shape={tuple(decoded.shape)}")
    print(f"core_loss={loss.item():.4f}")
    print(f"max_grad_norm={max(norm.item() for norm in grad_norms):.4f}")
    print("optimizer_step=PASS")
    print(f"retrieval_ranking={ranking.tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
