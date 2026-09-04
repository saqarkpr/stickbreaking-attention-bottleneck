"""
A stick-breaking variational bottleneck for pooling a sequence of vectors
into one: instead of a fixed pooling operator (mean, max, [CLS]), each token
gets a variational "vote" on how much it contributes, via a reparameterizable
stick-breaking process (a Kumaraswamy-distribution relaxation, which is what
makes stick-breaking differentiable end-to-end rather than requiring a
score-function estimator), and a Gaussian VAE-style posterior over what it
contributes. A single KL term regularizes both, exactly as in a standard
variational information bottleneck objective.

The central empirical question this module exists to test: does replacing
fixed pooling with this variational, unbounded-capacity mixture improve
generalization to sequences longer than anything seen in training? The
concentration of the mixture (how many tokens end up mattering) is controlled
by `alpha0`, and which direction of `alpha0` helps is *not* the intuitive one
-- see the comment on `self.alpha0` below and the alpha0 sweep in the README,
which found and corrected a wrong assumption about this made earlier in this
project's own development.

Design choices and their cost:
  - a nonparametric-flavoured (stick-breaking) prior over how many components
    are used, rather than a fixed pooling operator
  - a Gaussian VAE-style posterior over each component's *content*
  - a single KL term added to the task loss, as in a standard VIB objective
  - simplification: stick-breaking here is order-dependent on the input
    sequence (later positions can only receive what earlier ones leave
    behind), which is the mechanism the order-randomisation control in this
    project's README is built to diagnose
  - simplification: the bottleneck is applied once, at a pooling step, not
    across a full attention memory at every layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StickBreakingBottleneck(nn.Module):
    """Stick-breaking variational bottleneck over a sequence of vectors.

    Given per-token hidden states H (B, T, D), produces:
      - a pooled representation z (B, D): a weighted mixture of per-token
        Gaussian samples, where the mixture weights pi_1..pi_T come from a
        reparameterizable stick-breaking process (Kumaraswamy sticks).
      - a KL term (scalar) regularizing both the stick-breaking weights
        towards a Beta(1, alpha0) prior (whose CONCENTRATION, not sparsity,
        alpha0 controls -- larger alpha0 spreads mass over more sticks) and
        the per-token Gaussian posteriors towards N(0, I).
    """

    def __init__(self, d_model: int, alpha0: float = 5.0, eps: float = 1e-6,
                  shuffle_sticks: bool = False):
        super().__init__()
        self.shuffle_sticks = shuffle_sticks
        self.d_model = d_model
        # Prior Beta(1, alpha0) on each stick proportion v_k.
        #
        # E[v] = 1/(1 + alpha0) is the fraction of the REMAINING stick taken at
        # each break, so LARGER alpha0 takes SMALLER slices, the stick survives
        # more breaks, and mixture mass spreads over MORE positions.
        #
        # Larger alpha0 therefore makes the mixture MORE uniform, not sparser.
        # An earlier version of this comment said the opposite; the alpha0 sweep
        # in the README refuted it (accuracy at 8x training length rises
        # monotonically 0.486 -> 0.944 as alpha0 goes 1 -> 50).
        self.alpha0 = alpha0
        self.eps = eps

        # Kumaraswamy stick parameters (a, b > 0) per token, from hidden state
        self.stick_proj = nn.Linear(d_model, 2)
        # Gaussian posterior over each token's "value" vector
        self.mu_proj = nn.Linear(d_model, d_model)
        self.logvar_proj = nn.Linear(d_model, d_model)

    def _kumaraswamy_sticks(self, a, b):
        # sample v ~ Kumaraswamy(a, b) via inverse-CDF, reparameterizable
        u = torch.rand_like(a).clamp(self.eps, 1 - self.eps)
        v = (1 - (1 - u).pow(1.0 / b)).pow(1.0 / a)
        return v.clamp(self.eps, 1 - self.eps)

    def _stick_breaking_weights(self, v, pad_mask):
        # v: (B, T) stick proportions -> pi_k = v_k * prod_{j<k}(1-v_j)
        B, T = v.shape
        one_minus_v = (1 - v).clamp(min=self.eps)
        # cumulative product of (1-v_j) for j < k, exclusive
        log_1mv = torch.log(one_minus_v)
        cum_log = torch.cumsum(log_1mv, dim=1) - log_1mv  # exclusive cumsum
        pi = v * torch.exp(cum_log)
        if pad_mask is not None:
            pi = pi * pad_mask.float()
        pi = pi / (pi.sum(dim=1, keepdim=True) + self.eps)
        return pi

    @staticmethod
    def _kl_kumaraswamy_beta(a, b, alpha_prior, beta_prior=1.0, n_terms=10):
        # closed-form-ish KL(Kumaraswamy(a,b) || Beta(alpha_prior, beta_prior)),
        # derived via the Kumaraswamy distribution's known moment structure.
        euler_gamma = 0.5772156649015329
        kl = (a - alpha_prior) / a * (-euler_gamma - torch.digamma(b) - 1.0 / b)
        kl = kl + torch.log(a * b + 1e-10) - math_log(alpha_prior) - math_log(beta_prior)
        kl = kl - (b - 1) / b
        # truncated series for the Taylor expansion term
        taylor = torch.zeros_like(a)
        for m in range(1, n_terms + 1):
            taylor = taylor + 1.0 / (m + a * b) * beta_fn(m / a, b)
        kl = kl + (beta_prior - 1) * b * taylor
        return kl

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor = None):
        """
        h: (B, T, D) encoder hidden states
        pad_mask: (B, T) bool, True where real token
        returns: pooled (B, D), kl (scalar), pi (B, T) mixture weights (for inspection)
        """
        ab = F.softplus(self.stick_proj(h)) + self.eps  # (B, T, 2), keep a,b > 0
        a, b = ab[..., 0], ab[..., 1]

        v = self._kumaraswamy_sticks(a, b)
        if pad_mask is not None:
            v = torch.where(pad_mask, v, torch.zeros_like(v))

        if self.shuffle_sticks:
            # ORDER-RANDOMISED CONTROL.
            # Stick-breaking is order-dependent: pi_k = v_k * prod_{j<k}(1-v_j),
            # so later positions can only receive what earlier ones leave. That
            # breaks permutation invariance -- a fully order-invariant
            # construction would not have this issue, at the cost of being
            # considerably harder to make reparameterizable. Drawing a fresh
            # random stick order per example restores exchangeability *in
            # expectation* while changing nothing else, which isolates
            # order-dependence as the cause rather than merely correlating
            # with it.
            B, T = v.shape
            perm = torch.argsort(torch.rand(B, T, device=v.device), dim=1)
            inv = torch.argsort(perm, dim=1)
            v_perm = torch.gather(v, 1, perm)
            mask_perm = torch.gather(pad_mask, 1, perm) if pad_mask is not None else None
            pi_perm = self._stick_breaking_weights(v_perm, mask_perm)
            pi = torch.gather(pi_perm, 1, inv)   # back to original token order
        else:
            pi = self._stick_breaking_weights(v, pad_mask)

        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # (B, T, D) per-token value samples

        pooled = (pi.unsqueeze(-1) * z).sum(dim=1)  # (B, D)

        # KL for the Gaussian value posteriors, masked and averaged over real tokens
        kl_gauss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)  # (B, T)
        if pad_mask is not None:
            kl_gauss = kl_gauss * pad_mask.float()
            n_tokens = pad_mask.float().sum(dim=1).clamp(min=1)
        else:
            n_tokens = torch.full((h.size(0),), h.size(1), device=h.device, dtype=torch.float32)
        kl_gauss = (kl_gauss.sum(dim=1) / n_tokens).mean()

        # KL for the stick-breaking weights towards the Beta(1, alpha0) prior
        kl_stick = self._kl_kumaraswamy_beta(a, b, alpha_prior=1.0, beta_prior=self.alpha0)
        if pad_mask is not None:
            kl_stick = kl_stick * pad_mask.float()
        kl_stick = (kl_stick.sum(dim=1) / n_tokens).mean()

        kl = kl_gauss + kl_stick
        return pooled, kl, pi


def math_log(x):
    import math
    return math.log(x)


def beta_fn(x, y):
    # B(x,y) = Gamma(x)Gamma(y)/Gamma(x+y), computed via lgamma for stability
    return torch.exp(torch.lgamma(x) + torch.lgamma(torch.as_tensor(y, dtype=x.dtype, device=x.device))
                      - torch.lgamma(x + y))


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 4, 10, 32
    h = torch.randn(B, T, D)
    pad_mask = torch.ones(B, T, dtype=torch.bool)
    pad_mask[0, 7:] = False  # simulate one padded sequence

    layer = StickBreakingBottleneck(D, alpha0=5.0)
    pooled, kl, pi = layer(h, pad_mask)
    print("pooled:", pooled.shape, "kl:", kl.item())
    print("pi row 0 (padded seq, should sum to 1 over first 7):", pi[0].detach().numpy())
    print("pi row sums:", pi.sum(dim=1))
