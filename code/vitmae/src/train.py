import torch
import shutil
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from accelerate import Accelerator
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from PIL import Image
from transformers import ViTMAEConfig, ViTMAEForPreTraining

from dataset.load import load_dataset

def show_image(axs, image):
    axs.imshow(torch.clip((image * np.array([0.5]) + np.array([0.5])) * 255, 0, 255).int())
    axs.axis('off')

def extract_rgb(tensor):
    return tensor[:,:3, :, :]

def show_segmentation(axs, segmentation):
    segmentation = segmentation.squeeze(0).detach().cpu()
    img2 = np.zeros((224, 224, 3), dtype=np.uint8)

    lulc_tile = segmentation / (17-1) * 255

    img2[:,:,0] = lulc_tile
    img2[:,:,1] = lulc_tile
    img2[:,:,2] = lulc_tile

    rgb_data = img2[:, :, :3]
    rgb_data = (rgb_data).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_data)

    axs.imshow(rgb_image)
    axs.axis('off')


def eval(cfg, model, epoch, test_loader, device):
    test_batch = next(iter(test_loader))
    pixel_values = test_batch['base'].to(device)
    segmentation = test_batch['lulc_segmentation'].to(device)

    input = torch.cat([pixel_values, segmentation], dim=1)
    outputs = model(input)
    y = model.unpatchify(outputs.logits)
    y = extract_rgb(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    
    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 *4)
    mask = model.unpatchify(mask)
    mask = extract_rgb(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    pixel_values = pixel_values.detach().cpu()
    x = torch.einsum('nchw->nhwc', pixel_values)

    im_masked = x * (1 - mask)
    im_paste = x * (1 - mask) + y * mask
    batch_size = pixel_values.shape[0]

    fig, axes = plt.subplots(nrows=batch_size, ncols=5, figsize=(15, 3*batch_size))
    cols = ['Original', 'Segmentation', 'Masked', 'Reconstruction', 'Reconstruction + Visible']

    if batch_size == 1:
        for ax, col in zip(axes, cols):
            ax.set_title(col)

        for batch in range(batch_size):
            show_image(axes[0], x[batch])
            show_segmentation(axes[1], segmentation[batch])
            show_image(axes[2], im_masked[batch])
            show_image(axes[3], y[batch])
            show_image(axes[4], im_paste[batch])

    else:
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)

        for batch in range(batch_size):
            show_image(axes[batch, 0], x[batch])
            show_segmentation(axes[batch, 1], segmentation[batch])
            show_image(axes[batch, 2], im_masked[batch])
            show_image(axes[batch, 3], y[batch])
            show_image(axes[batch, 4], im_paste[batch])

    test_dir = os.path.join(cfg.train.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    
    fig.savefig(f"{test_dir}/{epoch:04d}.png")

def train(cfg, model, train_loader, test_loader, optimizer, accelerator, start_epoch=0):
    print("Starting Training")
    global_step = 0
    for epoch in range(start_epoch, cfg.train.num_epochs):
        model.train()

        progress_bar = tqdm(total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch}")

        total_loss = 0
        for batch in train_loader:

            satellite_images = batch['base'].to(device)
            segmentation = batch['lulc_segmentation'].to(device)

            input = torch.cat([satellite_images, segmentation], dim=1)

            outputs = model(input)

            loss = outputs.loss
            total_loss += loss.detach()

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()


            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}

            accelerator.log(logs, step=global_step)

            accelerator.log

            progress_bar.set_postfix(**logs)
            global_step += 1

        avg_loss = total_loss / len(train_loader)

        logs['avg_loss'] = avg_loss.item()
        progress_bar.set_postfix(**logs)

        if accelerator.is_main_process:
            model.eval()

            if (epoch + 1) % cfg.train.save_model_epochs == 0 or epoch == cfg.train.num_epochs - 1:
                save_dir = f"{cfg.train.output_dir}/checkpoint_{epoch}"
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)

                optimizer_state = optimizer.state_dict()
                optimizer_state["metadata"] = {"epoch": epoch}
                torch.save(optimizer_state, os.path.join(save_dir, "optimizer.pt"))

                with open(f"{cfg.train.output_dir}/params.yaml", "w") as f:
                    OmegaConf.save(cfg, f)
                
            if (epoch + 1) % cfg.train.save_image_epochs == 0 or epoch == cfg.train.num_epochs - 1:
                eval(cfg, model, epoch, test_loader, device)

if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    device = "mps"

    parser = argparse.ArgumentParser(description="PyTorch Training Script")
    parser.add_argument(
        "--checkpoint", type=str, help="Path to the checkpoint file", default=None
    )
    args = parser.parse_args()

    train_loader, test_loader = load_dataset(cfg)

    config = ViTMAEConfig(
        image_size=cfg.train.image_size,
        patch_size=cfg.train.patch_size,
        num_channels=cfg.train.num_channels,
        hidden_size=cfg.train.hidden_size,
        num_hidden_layers=cfg.train.num_hidden_layers,
        num_attention_heads=cfg.train.num_attention_heads,
        intermediate_size=cfg.train.intermediate_size,
        mask_ratio=cfg.train.mask_ratio
    )

    start_epoch = 0
    print(args.checkpoint)
    if args.checkpoint:

        print(f"Loading checkpoint from {args.checkpoint}...")
        model = ViTMAEForPreTraining.from_pretrained(args.checkpoint)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
        
        optimizer_state_path = os.path.join(args.checkpoint, "optimizer.pt")
        if os.path.exists(optimizer_state_path):
            optimizer_state = torch.load(optimizer_state_path)
            optimizer.load_state_dict(optimizer_state)
            print("Loaded optimizer state.")

        start_epoch = optimizer_state.get("metadata", {}).get("epoch", 0)
        print(f"Loaded optimizer state. Resuming from epoch {start_epoch}")

    else:
        print("No checkpoint found or provided. Starting training from scratch.")

        model = ViTMAEForPreTraining(config)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)


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

    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

    train(cfg, model, train_loader, test_loader, optimizer, accelerator, start_epoch)