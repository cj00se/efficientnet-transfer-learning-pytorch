from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from torchvision import transforms
import torchvision

        
def create_dataloaders(
        train_dir :str,
        test_dir: str,
        train_transform: transforms.Compose,
        test_transform: transforms.Compose,
        batch_size: int,
        num_workers: int):

    
    train_data = Dataset(train_dir, transform=train_transform)
    test_data = torchvision.utils.ImageFol(test_dir, transform=test_transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  pin_memory=True)
    
    return train_dataloader, test_dataloader, class_names
