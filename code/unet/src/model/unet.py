import torch
import torch.nn.functional as F
import torch.nn as nn
from diffusers import UNet2DModel, UNet2DConditionModel



class CustomUNet2DConditionModel(UNet2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.seg_embed = nn.Embedding(2, 1280).to('mps')


    def forward(self, sample, timestep, segmentation, **kwargs):
        # batch_size = segmentation.shape[0]
        # width = segmentation.shape[2]
        # height = segmentation.shape[3]

        # mask_flat = segmentation.view(batch_size, width * height)

        # mask_embeddings = self.seg_embed(mask_flat) 

        # return super().forward(sample=sample, timestep=timestep, encoder_hidden_states=mask_embeddings, **kwargs)
        return super().forward(sample=sample, timestep=timestep, **kwargs)
    


def load_unet_model(cfg):
    return CustomUNet2DConditionModel(
            sample_size=cfg.train.image_size,
            in_channels=4,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            )
        )