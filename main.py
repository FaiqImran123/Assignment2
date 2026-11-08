import argparse
from trainer import train_diffusion, visualize_diffusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "sample"], default="train")
    p.add_argument("--data_path", default="./data")
    p.add_argument("--train_steps", type=int, default=5000)
    p.add_argument("--fid", action="store_true", help="compute FID at each save")
    p.add_argument("--save_and_sample_every", type=int, default=1000)
    p.add_argument("--dataloader_workers", type=int, default=2)
    p.add_argument("--save_folder", default="./results/")
    p.add_argument("--load_path", default=None)
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--data_class", default="cat")
    p.add_argument("--time_steps", type=int, default=50)
    p.add_argument("--unet_dim", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    a = p.parse_args()

    if a.mode == "train":
        train_diffusion(
            data_path=a.data_path, train_steps=a.train_steps, fid=a.fid,
            save_and_sample_every=a.save_and_sample_every,
            dataloader_workers=a.dataloader_workers, save_folder=a.save_folder,
            load_path=a.load_path, image_size=a.image_size, batch_size=a.batch_size,
            data_class=a.data_class, time_steps=a.time_steps, unet_dim=a.unet_dim,
            learning_rate=a.learning_rate,
        )
    else:
        visualize_diffusion(
            load_path=a.load_path or "./results/model.pt", save_folder=a.save_folder,
            image_size=a.image_size, time_steps=a.time_steps, unet_dim=a.unet_dim,
        )


if __name__ == "__main__":
    main()
