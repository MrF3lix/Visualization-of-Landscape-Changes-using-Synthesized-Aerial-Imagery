
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import imageio.v3 as iio
from torch.utils.data import Dataset

class SwissImageDataset(Dataset):
    def __init__(self, base_dir, lulc_dir, lulc_gray_dir, lulc_segmentation_dir, alti_dir, transform=None):
        self.base_dir = base_dir
        self.lulc_dir = lulc_dir
        self.lulc_gray_dir = lulc_gray_dir
        self.lulc_segmentation_dir = lulc_segmentation_dir
        self.alti_dir = alti_dir
        self.transform = transform

        self.image_names = sorted(os.listdir(base_dir))
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        base_img_name = self.image_names[index]
        lulc_img_name = base_img_name.replace('.png', '.npy')
        # alti_img_name = base_img_name

        base_img = Image.open(os.path.join(self.base_dir, base_img_name)).convert('RGB')
        # lulc_img = Image.open(os.path.join(self.lulc_dir, lulc_img_name)).convert('RGB')
        # lulc_gray_img = Image.open(os.path.join(self.lulc_gray_dir, lulc_img_name)).convert('L')
        # lulc_segmentation = Image.open(os.path.join(self.lulc_segmentation_dir, lulc_img_name)).convert('L')
        # lulc_segmentation = iio.imread(os.path.join(self.lulc_segmentation_dir, lulc_img_name))

        lulc_segmentation = np.load(os.path.join(self.lulc_segmentation_dir, lulc_img_name))

        # alti_img = Image.open(os.path.join(self.alti_dir, alti_img_name)).convert('L')

        if self.transform:
            base_img = self.transform(base_img)
            # lulc_img = self.transform(lulc_img)
            # alti_img = self.transform(alti_img)

        transform = transforms.Compose([
            transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST_EXACT)
        ])

        # TODO: Reduce to minimal amount of classes

        null_seg = torch.zeros((40, 40))
        if lulc_segmentation.shape != null_seg.shape:
            lulc_segmentation = null_seg
        else:
            lulc_segmentation = torch.from_numpy(lulc_segmentation)
        
        lulc_segmentation = lulc_segmentation.unsqueeze(0).unsqueeze(0)

        # lulc_segmentation = lulc_segmentation.int()
        lulc_segmentation = transform(lulc_segmentation).int()
        lulc_segmentation = lulc_segmentation.squeeze(0)
        # lulc_segmentation = F.one_hot(lulc_segmentation, num_classes=72)
        # lulc_segmentation = lulc_segmentation.permute(2, 0, 1, 3)

        # return {'base': base_img, 'land_use_class': lulc_class, 'land_use': lulc_img, 'lulc_segmentation': lulc_segmentation, 'alti': alti_img}
        return {'base': base_img, 'lulc_segmentation': lulc_segmentation}
