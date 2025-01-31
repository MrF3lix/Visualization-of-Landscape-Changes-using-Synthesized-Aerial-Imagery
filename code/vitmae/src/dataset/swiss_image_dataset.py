
import os
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class SwissImageDataset(Dataset):
    def __init__(self, cfg, base_dir, lulc_segmentation_dir, transform=None):
        self.cfg = cfg
        self.base_dir = base_dir
        self.lulc_segmentation_dir = lulc_segmentation_dir
        self.transform = transform

        self.image_names = sorted(os.listdir(base_dir))
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        base_img_name = self.image_names[index]
        lulc_img_name = base_img_name.replace('.png', '.npy')

        try:
            base_img = Image.open(os.path.join(self.base_dir, base_img_name)).convert('RGB')
            lulc_segmentation = np.load(os.path.join(self.lulc_segmentation_dir, lulc_img_name))

            if self.transform:
                base_img = self.transform(base_img)

            transform = transforms.Compose([
                transforms.Resize((self.cfg.train.image_size, self.cfg.train.image_size), interpolation=transforms.InterpolationMode.NEAREST_EXACT)
            ])

            null_seg = torch.zeros((40, 40))
            if lulc_segmentation.shape != null_seg.shape:
                lulc_segmentation = null_seg
            else:
                lulc_segmentation = torch.from_numpy(lulc_segmentation)
            
            lulc_segmentation = lulc_segmentation.unsqueeze(0).unsqueeze(0)

            lulc_segmentation = transform(lulc_segmentation).int()
            lulc_segmentation = lulc_segmentation.squeeze(0)
            return {'base': base_img, 'lulc_segmentation': lulc_segmentation}
        except Exception as e:
            print(f'Could not load image: {base_img_name}')
            print(e)

