import os
import shutil
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from unet import UNet
from diffusion import Diffusion
from utils import AFHQDataset, cycle, find_image_dir, get_device


def build_model(image_size=32, time_steps=50, unet_dim=16, unet_dim_mults=(1, 2, 4, 8), device=None):
    device = device or get_device()
    unet = UNet(dim=unet_dim, dim_mults=tuple(unet_dim_mults), channels=3)
    diffusion = Diffusion(unet, image_size=image_size, time_steps=time_steps, loss_type="l1")
    return diffusion.to(device)


@torch.no_grad()
def generate_to_folder(diffusion, folder, n_images, batch_size=64):
    """Sample n_images and save them as PNGs in folder (used for FID)."""
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    diffusion.eval()
    count = 0
    while count < n_images:
        b = min(batch_size, n_images - count)
        imgs = ((diffusion.sample(b) + 1) / 2).clamp(0, 1)
        for img in imgs:
            save_image(img, os.path.join(folder, f"{count:05d}.png"))
            count += 1
    diffusion.train()


def compute_fid(real_dir, gen_dir, device):
    from cleanfid import fid as cleanfid
    return cleanfid.compute_fid(real_dir, gen_dir, mode="clean", num_workers=0,
                                device=torch.device(device), verbose=False)


def train_diffusion(
    data_path="./data",
    train_steps=5000,
    fid=True,
    save_and_sample_every=1000,
    dataloader_workers=2,
    save_folder="./results/",
    load_path=None,
    image_size=32,
    batch_size=32,
    data_class="cat",
    time_steps=50,
    unet_dim=16,
    learning_rate=1e-3,
    unet_dim_mults=(1, 2, 4, 8),
    num_fid_samples=500,
):
    device = get_device()
    os.makedirs(save_folder, exist_ok=True)

    img_dir = find_image_dir(data_path, data_class)
    ds = AFHQDataset(img_dir, image_size)
    print(f"Found {len(ds)} images in {img_dir}")
    dl = cycle(DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                          num_workers=dataloader_workers))

    diffusion = build_model(image_size, time_steps, unet_dim, unet_dim_mults, device)
    opt = Adam(diffusion.parameters(), lr=learning_rate)

    start = 0
    if load_path and os.path.exists(load_path):
        ckpt = torch.load(load_path, map_location=device)
        diffusion.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start = ckpt["step"]
        print(f"Resumed from step {start}")

    fid_history = []
    pbar = tqdm(range(start, train_steps), desc="training")
    for step in pbar:
        x0 = next(dl).to(device)
        loss = diffusion(x0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        s = step + 1
        if s % save_and_sample_every == 0 or s == train_steps:
            torch.save({"model": diffusion.state_dict(), "opt": opt.state_dict(), "step": s},
                       os.path.join(save_folder, "model.pt"))
            diffusion.eval()
            with torch.no_grad():
                imgs = ((diffusion.sample(16) + 1) / 2).clamp(0, 1)
            diffusion.train()
            save_image(imgs, os.path.join(save_folder, f"sample-{s}.png"), nrow=4)
            if fid:
                gen_dir = os.path.join(save_folder, "fid_samples")
                generate_to_folder(diffusion, gen_dir, num_fid_samples)
                score = compute_fid(img_dir, gen_dir, device)
                fid_history.append((s, score))
                print(f"\nstep {s}: FID = {score:.2f}")

    if fid_history:
        print("FID history:", fid_history)
    return diffusion


def visualize_diffusion(
    load_path="./results/model.pt",
    save_folder="./results/",
    image_size=32,
    time_steps=50,
    unet_dim=16,
    unet_dim_mults=(1, 2, 4, 8),
    n_images=16,
):
    """Load a trained model and save a grid of generated images."""
    device = get_device()
    os.makedirs(save_folder, exist_ok=True)
    diffusion = build_model(image_size, time_steps, unet_dim, unet_dim_mults, device)
    ckpt = torch.load(load_path, map_location=device)
    diffusion.load_state_dict(ckpt["model"])
    diffusion.eval()
    with torch.no_grad():
        imgs = ((diffusion.sample(n_images) + 1) / 2).clamp(0, 1)
    out = os.path.join(save_folder, "generated.png")
    save_image(imgs, out, nrow=4)
    print("Saved", out)
    return out
