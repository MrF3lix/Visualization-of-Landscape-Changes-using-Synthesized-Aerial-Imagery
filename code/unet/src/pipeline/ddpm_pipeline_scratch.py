import torch

class DDPMPipeline:
    def __init__(self, model, noise_scheduler, device="cpu"):
        self.model = model.to(device)
        self.noise_scheduler = noise_scheduler
        self.device = device

    def generate(self, shape, num_timesteps, segmentation_map=None):
        with torch.no_grad():
            img = torch.randn(shape, device=self.device)
            # img = torch.cat([img, segmentation_map], dim=1)

            for t in reversed(range(num_timesteps)):
                timesteps = torch.full((shape[0],), t, device=self.device, dtype=torch.float32)

                predicted_noise = self.model(torch.cat([img, segmentation_map], dim=1), timesteps)

                alpha_t = self.noise_scheduler.alphas_cumprod[t]
                alpha_t_prev = self.noise_scheduler.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
                beta_t = self.noise_scheduler.betas[t]

                mean = (img - beta_t * predicted_noise / torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_t)

                if t > 0:
                    noise = torch.randn_like(img)
                    variance = beta_t * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t))
                    img = mean + variance * noise
                else:
                    img = mean

        return img