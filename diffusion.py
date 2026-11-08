import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def gather_timestep_coeff(coeff, t, x_shape):
    """Extract coeff[t] for a batch and reshape to (B, 1, 1, 1) for broadcasting."""
    b = t.shape[0]
    out = coeff.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_alphas(time_steps, s=0.008):
    """Cosine schedule (Eq. 14). Returns alpha_t for t = 0..T-1 (length T)."""
    steps = torch.arange(time_steps + 1, dtype=torch.float64)
    f = torch.cos(((steps / time_steps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = f / f[0]
    alphas = torch.clip(alphas_bar[1:] / alphas_bar[:-1], 0.001, 1.0)
    return alphas.float()


class Diffusion(nn.Module):
    def __init__(self, model, image_size, time_steps=50, loss_type="l1",
                 channels=3, s=0.008):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.time_steps = time_steps
        self.loss_type = loss_type

        # ---- schedule coefficients ----
        alphas = cosine_alphas(time_steps, s)                       # alpha_t
        alphas_cumprod = torch.cumprod(alphas, dim=0)               # alpha_bar_t
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # alpha_bar_{t-1}

        def reg(name, val):
            self.register_buffer(name, val.float())

        reg("alphas", alphas)
        reg("alphas_cumprod", alphas_cumprod)
        reg("alphas_cumprod_prev", alphas_cumprod_prev)

        # forward process q(x_t | x_0)
        reg("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        reg("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # x0 estimate: x0 = (x_t - sqrt(1-abar) * eps) / sqrt(abar)
        reg("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))

        # posterior q(x_{t-1} | x_t, x_0)  (Eq. 18)
        reg("posterior_mean_coef1",
            torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        reg("posterior_mean_coef2",
            torch.sqrt(alphas_cumprod_prev) * (1.0 - alphas) / (1.0 - alphas_cumprod))
        posterior_variance = (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) * (1.0 - alphas)
        reg("posterior_variance", posterior_variance)

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------
    def q_sample(self, x0, t, noise=None):
        """x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps  (Eq. 13)."""
        if noise is None:
            noise = torch.randn_like(x0)
        c1 = gather_timestep_coeff(self.sqrt_alphas_cumprod, t, x0.shape)
        c2 = gather_timestep_coeff(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return c1 * x0 + c2 * noise

    def p_losses(self, x0, t, noise=None):
        """L1 loss between true noise and predicted noise (Alg. 1)."""
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = self.model(x_t, t)
        if self.loss_type == "l1":
            return F.l1_loss(pred_noise, noise)
        elif self.loss_type == "l2":
            return F.mse_loss(pred_noise, noise)
        raise NotImplementedError(self.loss_type)

    def forward(self, x0):
        """Training step: sample t ~ Uniform({0..T-1}) and return the loss."""
        b = x0.shape[0]
        t = torch.randint(0, self.time_steps, (b,), device=x0.device).long()
        return self.p_losses(x0, t)

    # ------------------------------------------------------------------
    # Reverse process
    # ------------------------------------------------------------------
    @torch.no_grad()
    def p_sample(self, x_t, t_int):
        """One reverse step x_t -> x_{t-1} (Alg. 2). t_int is a python int."""
        b = x_t.shape[0]
        t = torch.full((b,), t_int, device=x_t.device, dtype=torch.long)

        eps = self.model(x_t, t)
        x0_hat = gather_timestep_coeff(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t \
            - gather_timestep_coeff(self.sqrt_recip_alphas_cumprod * self.sqrt_one_minus_alphas_cumprod,
                                    t, x_t.shape) * eps
        x0_hat = x0_hat.clamp(-1.0, 1.0)

        mean = gather_timestep_coeff(self.posterior_mean_coef1, t, x_t.shape) * x_t \
            + gather_timestep_coeff(self.posterior_mean_coef2, t, x_t.shape) * x0_hat
        var = gather_timestep_coeff(self.posterior_variance, t, x_t.shape)

        if t_int > 0:
            z = torch.randn_like(x_t)
        else:
            z = torch.zeros_like(x_t)
        return mean + torch.sqrt(var) * z

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """Start from pure noise and run the full reverse chain."""
        device = self.alphas.device
        x = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(self.time_steps)), desc="sampling",
                      total=self.time_steps, leave=False):
            x = self.p_sample(x, t)
        return x

    @torch.no_grad()
    def sample(self, batch_size=16):
        """Generate images. Output is in [-1, 1]; the training utils typically
        rescale to [0, 1] via (x + 1) / 2 -- do this here if your starter's
        trainer/utils expect it."""
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        return self.p_sample_loop(shape)
