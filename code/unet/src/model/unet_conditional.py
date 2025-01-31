import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import ModelMixin

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

class UNetConditional(ModelMixin):
    def __init__(self, in_channels=3, out_channels=3, features=[128, 256, 512, 1024]):
        super(UNetConditional, self).__init__()

        self.config_name = 'UNetConditionalScratch'

        self.timestep_embedding = nn.Linear(1, features[0])

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.downs_conditions = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            # self.downs.append(DoubleConv(in_channels, feature // 2))
            self.downs.append(DoubleConv(in_channels, feature))
            self.downs_conditions.append(DoubleConv(in_channels // 2 if in_channels != 4 else 1, feature // 2)) # TODO: Remove this hack
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )

            self.ups.append(DoubleConv(feature*2, feature))

        # self.bottleneck = DoubleConv(features[-1]//2, features[-1]*2)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def forward(self, x, t, condition):

        t = t.float().unsqueeze(-1)  # Ensure t has shape (batch_size, 1)
        t_emb = self.timestep_embedding(t).unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, base_channels, 1, 1)

        skip_connections = []

        # Applying condition only on the first down block
        # Adds one layer to the image with the corresponding segementation map
        # x = torch.cat([x, condition], dim=1)

        for down, down_condition in zip(self.downs, self.downs_conditions):
            # x = torch.cat([x, condition], dim=1)

            # x = torch.cat([x, F.interpolate(condition, size=x.shape[2:])], dim=1)
            # x = torch.cat([x, condition], dim=1)

            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

            condition = down_condition(condition)
            condition = self.pool(condition)


        # x = torch.cat([x, condition], dim=1)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)


        x = x + t_emb # TODO: Use Concat instead?

        return self.final_conv(x)
    


# Example usage
if __name__ == "__main__":
    model = UNetConditional(in_channels=3, out_channels=3)
    input_tensor = torch.randn(8, 3, 128, 128)  # Batch size of 8, 3 input channels, 128x128 image
    condition_tensor = torch.randn(8, 1, 128, 128)  # Condition
    timesteps = torch.randn(8)  # 1D tensor for timesteps
    output = model(input_tensor, timesteps, condition=condition_tensor)
    print(output.shape)  # Should be (8, 3, 128, 128)
