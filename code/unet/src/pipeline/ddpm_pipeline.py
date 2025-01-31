
import torch
from diffusers import DDPMPipeline

class ConditionalDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler):
        super().__init__(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, land_use_map):
        batch_size = land_use_map.shape[0]

        noise = torch.randn(batch_size, 3, land_use_map.shape[2], land_use_map.shape[3]).to(land_use_map.device)
        x = noise

        for i, t in enumerate(self.scheduler.timesteps):
            model_output = self.unet(x, t.to('mps'), condition=land_use_map)

            # model_output = model_output["sample"]
            x = self.scheduler.step(model_output, t, x)["prev_sample"]

        generated_image = x[:, :3, :, :]
        return generated_image
