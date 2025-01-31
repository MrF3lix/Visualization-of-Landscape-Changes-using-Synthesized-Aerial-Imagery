import os
import torch
from omegaconf import OmegaConf
from diffusers import DDPMScheduler
from diffusers import DiffusionPipeline
from model.unet import UNet
from dataset.load import load_dataset
from pipeline.ddpm_pipeline import ConditionalDDPMPipeline
import matplotlib.pyplot as plt

def evaluate(cfg, pipeline, test_loader):
    batch = next(iter(test_loader))

    images = pipeline(
        land_use_map=batch['land_use'].to('mps')
    )

    print(images.shape)
    print(batch['land_use'].shape)
    print(batch['base'].shape)

    for i in range(len(images)):
        generated_image = images[i]
        generated_image_np = generated_image.cpu().permute(1, 2, 0).numpy()
        test_dir = os.path.join(cfg.train.output_dir, "samples")

        landuse_image_np = batch['land_use'][i].permute(1, 2, 0).numpy()
        base_image_np = batch['base'][i].permute(1, 2, 0).numpy()

        os.makedirs(test_dir, exist_ok=True)
        plt.imsave(f"{test_dir}/test_{i:02d}.png", (generated_image_np * 255).astype("uint8"))
        plt.imsave(f"{test_dir}/test_{i:02d}_landuse.png", (landuse_image_np * 255).astype("uint8"))
        plt.imsave(f"{test_dir}/test_{i:02d}_base.png", (base_image_np * 255).astype("uint8"))


def test(cfg):
    _, test_loader = load_dataset(cfg)

    pipeline = ConditionalDDPMPipeline.from_pretrained(cfg.train.output_dir).to('mps')

    evaluate(cfg, pipeline, test_loader)
    
    return

if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    test(cfg)