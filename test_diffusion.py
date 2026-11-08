import torch
from unet import UNet
from diffusion import Diffusion


def make(T=50):
    return Diffusion(UNet(dim=16, dim_mults=(1, 2, 4, 8)), image_size=32, time_steps=T)


def test_unet_shape():
    net = UNet(dim=16, dim_mults=(1, 2, 4, 8))
    out = net(torch.randn(4, 3, 32, 32), torch.randint(0, 50, (4,)))
    assert out.shape == (4, 3, 32, 32)
    print("UNet output shape OK")


def test_schedule():
    d = make()
    ab = d.alphas_cumprod
    assert (ab[1:] <= ab[:-1]).all(), "alpha_bar must decrease"
    assert ab[0] > 0.9 and ab[-1] < 0.05
    assert d.posterior_variance[0] == 0
    assert (d.alphas >= 0.001).all() and (d.alphas <= 1).all()
    print("Schedule OK")


def test_q_sample():
    d = make()
    x0 = torch.randn(4, 3, 32, 32)
    noise = torch.randn_like(x0)
    t = torch.zeros(4, dtype=torch.long)
    assert (d.q_sample(x0, t, noise) - x0).abs().max() < 0.3   # t=0 is almost clean
    t = torch.full((4,), 49, dtype=torch.long)
    xt = d.q_sample(x0, t, noise)
    assert (xt - noise).abs().max() < 0.5                       # t=T-1 is almost noise
    print("q_sample OK")


def test_loss_backward():
    d = make()
    loss = d(torch.randn(4, 3, 32, 32))
    assert loss.ndim == 0
    loss.backward()
    assert any(p.grad is not None for p in d.parameters())
    print("Loss + backward OK, loss =", float(loss))


def test_sample():
    d = make(T=5)
    d.eval()
    out = d.sample(2)
    assert out.shape == (2, 3, 32, 32)
    assert out.abs().max() <= 1.0 + 1e-5
    print("Sampling OK")


if __name__ == "__main__":
    test_unet_shape()
    test_schedule()
    test_q_sample()
    test_loss_backward()
    test_sample()
    print("\nAll tests passed.")
