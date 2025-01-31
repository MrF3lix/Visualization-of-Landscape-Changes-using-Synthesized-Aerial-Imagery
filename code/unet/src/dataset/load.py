import torchvision.transforms as transforms
from dataset.swiss_image_dataset import SwissImageDataset
from torch.utils.data import DataLoader, Subset, random_split

def load_dataset(cfg):
    transform = transforms.Compose([
        transforms.Resize((cfg.train.image_size, cfg.train.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = SwissImageDataset(
        base_dir=f"{cfg.train.dataset_base_path}/base",
        lulc_dir=f"{cfg.train.dataset_base_path}/lulc",
        lulc_gray_dir=f"{cfg.train.dataset_base_path}/lulc_gray",
        lulc_segmentation_dir=f"{cfg.train.dataset_base_path}/lulc_segmentation_forest_npy",
        alti_dir=f"{cfg.train.dataset_base_path}/alti",
        transform=transform
    )

    indices = list(range(0,cfg.train.subset_size)) 
    subset = Subset(dataset, indices)
 
    dataset_len = len(subset)
    train_len = int(dataset_len*cfg.train.train_test_split)       
    train_set, test_set = random_split(subset, [train_len, dataset_len-train_len])
    
    train_loader = DataLoader(train_set, batch_size=cfg.train.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg.train.test_batch_size, shuffle=False)

    return train_loader, test_loader
