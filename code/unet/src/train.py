import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator, notebook_launcher
from omegaconf import OmegaConf
from dataset.load import load_dataset
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from model.unet import load_unet_model
from pipeline.ddpm_pipeline import ConditionalDDPMPipeline
from pipeline.ddim_pipeline import ConditionalDDIMPipeline
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

invTrans = transforms.Compose([ 
    transforms.Normalize(mean = [ 0., 0., 0. ],
    std = [ 1/0.5, 1/0.5, 1/0.5 ]),
    transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
    std = [ 1., 1., 1. ]),
])

def train_model(cfg, model, optimizer, lr_scheduler, noise_scheduler, train_dataloader, test_dataloader, device):  
    accelerator = Accelerator(
        mixed_precision=cfg.train.mixed_precision,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(cfg.train.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        if cfg.train.output_dir is not None:
            os.makedirs(cfg.train.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)
    global_step = 0

    for epoch in range(cfg.train.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            satellite_images = batch['base'].to(device)
            segmented_images = batch['lulc_segmentation'].to(device)
            
            t = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (satellite_images.shape[0],), device=satellite_images.device,
                dtype=torch.int64
            )

            noise = torch.randn_like(satellite_images, device=satellite_images.device)
            x_noisy = noise_scheduler.add_noise(satellite_images, noise, t)

            # TODO: Add masked input image?
            # mask = torch.randint(0, 1, (8,8))
            x_input = torch.cat([x_noisy, segmented_images], dim=1)

            noise_pred = model(x_input, t, segmentation=segmented_images)
            loss = F.mse_loss(noise_pred[0], noise)
            
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        
        if accelerator.is_main_process:
            model.eval()
            pipeline = ConditionalDDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % cfg.train.save_model_epochs == 0 or epoch == cfg.train.num_epochs - 1:
                pipeline.save_pretrained(cfg.train.output_dir)
                
            if (epoch + 1) % cfg.train.save_image_epochs == 0 or epoch == cfg.train.num_epochs - 1:
                evaluate(cfg, epoch, pipeline, test_dataloader, device)
    
def evaluate(cfg, epoch, pipeline, test_loader, device):
    batch = next(iter(test_loader))

    images = pipeline(
        base_img=batch['base'].to(device),
        land_use_map=batch['lulc_segmentation'].to(device),
        generator=torch.Generator(device=device).manual_seed(cfg.train.seed)
    )

    for i in range(len(images)):
        generated_image = images[i]
        generated_image_np = generated_image.cpu()
        test_dir = os.path.join(cfg.train.output_dir, "samples")

        # landuse_image_np = batch['land_use'][i]
        base_image_np = batch['base'][i]

        generated_image_np = invTrans(generated_image_np).permute(1, 2, 0).numpy()
        # landuse_image_np = invTrans(landuse_image_np).permute(1, 2, 0).numpy()
        base_image_np = invTrans(base_image_np).permute(1, 2, 0).numpy()

        os.makedirs(test_dir, exist_ok=True)

        save_as17_tile(batch['lulc_segmentation'][i], test_dir, i)

        plt.imsave(f"{test_dir}/{i:02d}_{epoch:04d}.png", (generated_image_np * 255).astype("uint8"))
        # plt.imsave(f"{test_dir}/{i:02d}_landuse.png", (landuse_image_np * 255).astype("uint8"))
        plt.imsave(f"{test_dir}/{i:02d}_base.png", (base_image_np * 255).astype("uint8"))

def save_as17_tile(lulc_tile, test_dir, i):
    lulc_tile = lulc_tile.squeeze(0)
    img2 = np.zeros( ( np.array(lulc_tile).shape[0], np.array(lulc_tile).shape[1], 3 ) )

    lulc_tile = lulc_tile / 2 * 255

    # print(lulc_tile)

    img2[:,:,0] = lulc_tile
    img2[:,:,1] = lulc_tile
    img2[:,:,2] = lulc_tile

    rgb_data = img2[:, :, :3]
    rgb_data = (rgb_data).astype(np.uint8)

    # rgb_image = Image.fromarray(rgb_data).resize((500, 500), Image.NEAREST)
    rgb_image = Image.fromarray(rgb_data)
    rgb_image.save(f"{test_dir}/{i:02d}_landuse_as17.png")


def train(device):
    train_loader, test_loader = load_dataset(cfg)

    model = load_unet_model(cfg)
    # noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    noise_scheduler = DDIMScheduler()
    
    noise_scheduler.set_timesteps(1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.train.lr_warmup_steps,
        num_training_steps=(len(train_loader) * cfg.train.num_epochs),
    )

    args = (cfg, model, optimizer, lr_scheduler, noise_scheduler, train_loader, test_loader, device)
    notebook_launcher(train_model, args, num_processes=1)
    return

if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    train(device)