

    

import torch

from torchvision import transforms
from pathlib import Path


from data_setup import create_dataloaders
from model_builder import FoodModel
import engine
import utils

NUM_EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"
train_dir = Path("C:/Python Latop/putting_all_together_with_FOOD101/data/train")
test_dir = Path("C:/Python Latop/putting_all_together_with_FOOD101/data/validation")
def convert_to_rgb(image):
    return image.convert("RGB")

if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.Resize(size=(224, 224)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor()
    ])


    train_dataloader, test_dataloader, class_names = (
        create_dataloaders(train_dir=train_dir,
                        test_dir=test_dir,
                        train_transform= train_transform,
                        test_transform = test_transform,
                        batch_size=BATCH_SIZE,
                        num_workers=8))

    model = FoodModel(input_size=3,
                        output_size= len(class_names)).to(device=device)
    model.load_state_dict(torch.load("models\\05_going_modular_script_mode.pth",
                                  map_location=device, weights_only=True))
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=NUM_EPOCHS,
                device=device)

    utils.save_model(model=model,
                    target_dir="models",
                    model_name="05_going_modular_script_mode.pth")