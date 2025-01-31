
import torch
from tqdm.auto import tqdm
from diffusers import DDIMPipeline

class ConditionalDDIMPipeline(DDIMPipeline):
    def __init__(self, unet, scheduler):
        super().__init__(unet=unet, scheduler=scheduler)

    def add_masked_noise(self, base_img, mask, noise):
        # neg_mask = (~mask).int().squeeze(0)
        # mask = (mask).int().squeeze(0)

        # masked_noise = noise * neg_mask

        # masked_image = base_img * mask
        # masked_image = masked_image + masked_noise

        # return masked_image

        return noise


    @torch.no_grad()
    def __call__(self, base_img, land_use_map, generator):

        batch_size = land_use_map.shape[0]

        # noise = torch.randn_like(land_use_map, device=land_use_map.device)

        # noise = torch.randn(batch_size, 3, 128, 128).to(land_use_map.device)
        # x = noise

        noise = torch.rand(batch_size, 3, 128, 128).to(base_img.device)
        x = noise.to(land_use_map.device)

        # x = self.add_masked_noise(base_img, land_use_map, noise).to(land_use_map.device)

        # TODO: Instead of nooise choose the input image

        # TODO: Apply the segmentation map to the noise

        # TODO: Add the masked noise to the image => 

        self.scheduler.set_timesteps(500)
        progress_bar = tqdm(total=1000)

        for i, t in enumerate(self.scheduler.timesteps):
            progress_bar.set_description(f"Steps {i}")
            
            x = self.add_masked_noise(base_img, land_use_map, x)
            x_input = torch.cat([x, land_use_map], dim=1)
            model_output = self.unet(x_input, segmentation=land_use_map, timestep=t)["sample"]
            x = self.scheduler.step(model_output, t, x, generator=generator)["prev_sample"]


            progress_bar.update(1)

        generated_image = x[:, :3, :, :]
        return generated_image
