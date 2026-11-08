# DDPM on AFHQ Cats

**Generative AI & LLM, Assignment 2 (IML Lab)**
**Author:** Faiq Imran (MSDS25007), ITU

A from-scratch implementation of a Denoising Diffusion Probabilistic Model (DDPM) that generates cat images. A U-Net is trained to predict the noise added during the forward diffusion process, and new images are produced by running the learned reverse process from pure Gaussian noise.

---

## Folder Layout

```
hw2/
├── data/
│   └── cat/                  <- put the AFHQ cat images here (png/jpg)
├── diffusion.py              <- forward/reverse diffusion, cosine noise schedule
├── unet.py                   <- U-Net noise-prediction network (with skip connections)
├── utils.py                  <- helper functions (image saving, sampling grids, etc.)
├── trainer.py                <- training loop, periodic sampling and checkpointing
├── main.py                   <- entry point: builds the model and starts training
├── test_diffusion.py         <- sanity tests for the diffusion implementation
├── requirements.txt          <- Python dependencies
├── README.md                 <- this file
├── (weights file)            <- downloaded from the link in "Pretrained Weights" below
└── Assignment2_Faiq_Imran_MSDS25007.pdf   <- written report and answers
```

Generated sample grids (e.g. `sample_0.png` ... `sample_9.png`) are written during training, one per save step.

---

## File Descriptions

| File | Purpose |
|---|---|
| `diffusion.py` | Cosine variance schedule ($s = 0.008$), forward noising $q(x_t \mid x_0)$ in closed form, and the reverse (denoising) sampling loop. |
| `unet.py` | U-Net that takes a noisy image and the timestep and predicts the noise. Skip connections preserve fine spatial detail. |
| `utils.py` | Small helpers shared by the other files. |
| `trainer.py` | Optimisation loop with the L1 noise-prediction loss, saving samples every N steps. |
| `main.py` | Sets the hyperparameters, loads the data from `data/cat/`, and runs training. |
| `test_diffusion.py` | Quick checks that the diffusion code behaves as expected. |

---

## Setup

```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt
```

**Data:** download the AFHQ dataset and copy the cat images (a subset is enough) into `data/cat/`.

**Pretrained weights:** download the trained model weights from Google Drive:
[Download weights](https://drive.google.com/file/d/1s90q5sGBi-G1t7jg6C6N_hzBD4N3oL1Z/view?usp=drive_link)

---

## Usage

### 1. Run the sanity tests

```bash
python test_diffusion.py
```

### 2. Train the model

```bash
python main.py
```

Training settings are defined in `main.py` / `trainer.py`. The two runs reported in the assignment used:

| Setting | Short run | Full run |
|---|---|---|
| Training steps | 1,000 | 10,000 |
| Save & sample every | 100 steps | 1,000 steps |
| Learning rate | 1e-3 | 1e-3 |
| Batch size | 32 | 32 |
| Loss | L1 | L1 |
| Noise schedule | Cosine ($s = 0.008$) | Cosine ($s = 0.008$) |
| Diffusion timesteps | about 50 | about 50 |
| FID calculation | Disabled | Disabled |

Change the number of steps and the save interval in the training configuration to reproduce either run.

---

## Pretrained Weights

Trained model weights (10,000-step run): [Google Drive link](https://drive.google.com/file/d/1s90q5sGBi-G1t7jg6C6N_hzBD4N3oL1Z/view?usp=drive_link)

Download the file and place it in the project folder (`hw2/`) so the code can load it.

---

## Results Summary

| Run | Final training loss | Sample quality |
|---|---|---|
| 1,000 steps | 0.166 (from about 1.5) | Blurry, no fine detail; rough cat shapes start to appear |
| 10,000 steps | about 0.129 | Clearly recognisable cats with eyes, nose, mouth, fur texture and ear shapes |

Expected runtime for the full 10,000-step run is about **40 minutes on a Colab T4 GPU**.

The forward process turns a cat image into pure Gaussian noise by about 99% of the timesteps, and the reverse process rebuilds a clear cat from that noise. See the report for the step-by-step visualisations.

---

## Notes

- A subset of AFHQ and limited compute were used, so image quality is below state of the art. The goal is to validate the core DDPM ideas.
- FID was disabled in both runs to keep training fast.
- The full written answers (CNNs, Transformers, GANs, VAEs and the diffusion derivations) are in the PDF report.