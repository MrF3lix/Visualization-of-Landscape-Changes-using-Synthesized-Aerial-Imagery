import torch
import torchvision.transforms.v2 as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from PIL import Image
from transformers import ViTMAEForPreTraining
from torch.utils.data import DataLoader

from dataset.swiss_image_dataset import SwissImageDataset

def load_dataset(cfg):
    transform = transforms.Compose([
        transforms.Resize((cfg.train.image_size, cfg.train.image_size)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = SwissImageDataset(
        cfg=cfg,
        base_dir=f"{cfg.eval.dataset_base_path}/base",
        lulc_segmentation_dir=f"{cfg.eval.dataset_base_path}/lulc_segmentation_as17_npy",
        transform=transform
    )

    return DataLoader(dataset, batch_size=cfg.eval.eval_batch_size, shuffle=True)


def show_image(axs, image):
    axs.imshow(torch.clip((image * np.array([0.5]) + np.array([0.5])) * 255, 0, 255).int())
    axs.axis('off')

def extract_rgb(tensor):
    return tensor[:,:3, :, :]

def extract_seg(tensor):
    return tensor[:,3:4, :, :]


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

    # TODO: Load new segmenttion
    # TODO: Load new Mask

    input = torch.cat([pixel_values, segmentation], dim=1)
    outputs = model(input)

    y = model.unpatchify(outputs.logits)
    y_seg = extract_seg(y)
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

    fig, axes = plt.subplots(nrows=batch_size, ncols=6, figsize=(18, 3*batch_size))
    cols = ['Original', 'Segmentation', 'Masked', 'Reconstruction', 'Reconstruction + Visible', 'Seg Recon']

    if batch_size == 1:
        for ax, col in zip(axes, cols):
            ax.set_title(col)

        for batch in range(batch_size):
            show_image(axes[0], x[batch])
            show_segmentation(axes[1], segmentation[batch])
            show_image(axes[2], im_masked[batch])
            show_image(axes[3], y[batch])
            show_image(axes[4], im_paste[batch])
            show_segmentation(axes[5], y_seg[batch])

    else:
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)

        for batch in range(batch_size):
            show_image(axes[batch, 0], x[batch])
            show_segmentation(axes[batch, 1], segmentation[batch])
            show_image(axes[batch, 2], im_masked[batch])
            show_image(axes[batch, 3], y[batch])
            show_image(axes[batch, 4], im_paste[batch])
            show_segmentation(axes[batch, 5], y_seg[batch].squeeze(1))

    test_dir = os.path.join(cfg.eval.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    
    fig.savefig(f"{test_dir}/{epoch:04d}.png")

if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = load_dataset(cfg)

    model = ViTMAEForPreTraining.from_pretrained(cfg.eval.checkpoint)

    eval(cfg, model, 1, data_loader, device)