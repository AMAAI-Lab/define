import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def masked_mean(feats, lengths):
    mask = (torch.arange(feats.shape[1], device=feats.device)[None, :] < lengths[:, None]).to(feats.dtype)[..., None]
    return (feats * mask).sum(1) / mask.sum(1).clamp(min=1.0)


# Supervised contrastive loss over accent labels.
def supcon_loss(z, labels, temperature=0.1):
    z = F.normalize(z, dim=-1)
    eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    sim = (z @ z.t() / temperature).masked_fill(eye, float("-inf"))
    pos = (labels[:, None] == labels[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n_pos = pos.sum(1)
    valid = n_pos > 0
    if not valid.any():
        return z.new_zeros(())
    return (-(logp.masked_fill(~pos, 0.0).sum(1)[valid] / n_pos[valid])).mean()


class AccentBottleneck(nn.Module):
    def __init__(self, in_dim=1024, hidden=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

    def forward(self, pooled):
        return F.normalize(self.net(pooled), dim=-1)


class AccentEncoder(nn.Module):
    # SSL frame features (B, T, 1024) + lengths -> unit accent vector (B, out_dim).
    def __init__(self, in_dim=1024, hidden=512, out_dim=128, n_speakers=0, grl_lambda=1.0):
        super().__init__()
        self.bottleneck = AccentBottleneck(in_dim, hidden, out_dim)
        self.grl_lambda = grl_lambda
        self.spk_head = nn.Linear(out_dim, n_speakers) if n_speakers > 0 else None

    def forward(self, feats, frame_lens):
        return self.bottleneck(masked_mean(feats, frame_lens))

    # Auxiliary losses: accent supcon and adversarial speaker classification.
    def losses(self, a, accent_labels, speaker_labels):
        out = {"supcon": supcon_loss(a, accent_labels)}
        if self.spk_head is not None:
            out["grl_spk"] = F.cross_entropy(self.spk_head(GradReverse.apply(a, self.grl_lambda)), speaker_labels)
        return out


# Average K crop embeddings per item and renormalise: (B*K, D) -> (B, D).
def pool_exemplars(a, k):
    return F.normalize(a.view(-1, k, a.shape[-1]).mean(1), dim=-1)
