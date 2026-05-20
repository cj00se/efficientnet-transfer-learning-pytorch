# %%
import torch 
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

import matplotlib.pyplot as plt

from torchinfo import summary


# %%
data_path_10 = Path("C:\Python Latop\experimenting\pizz_steak_sushi_10")
data_path_20 =  Path("C:\Python Latop\experimenting\pizza_steak_sushi_20")

data_path_10_train = data_path_10 / "train"
data_path_10_test = data_path_10 / "test"

data_path_20_train = data_path_20 / "train"
data_path_20_test = data_path_20 / "test"


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"


# %%
def create_dataloaders(image_size, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_10 = datasets.ImageFolder(data_path_10_train, transform=transform)
    test_10  = datasets.ImageFolder(data_path_10_test,  transform=transform)
    train_20 = datasets.ImageFolder(data_path_20_train, transform=transform)
    test_20  = datasets.ImageFolder(data_path_20_test,  transform=transform)
    
    return (DataLoader(train_10, batch_size=batch_size, shuffle=True),
            DataLoader(test_10,  batch_size=batch_size, shuffle=False),
            DataLoader(train_20, batch_size=batch_size, shuffle=True),
            DataLoader(test_20,  batch_size=batch_size, shuffle=False))

# %%
def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (tensor * std + mean).clamp(0, 1)

# %%
image, label = next(iter(train_10_dataloader))
plt.imshow(unnormalize(image[30]).permute(1,2,0))

# %%


# %%
def create_effnetb0():

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights = weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(p=.02, inplace=True),
        nn.Linear(in_features=1280, out_features=3, bias=True)
    )

    return model


def create_effnetb2():

    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights = weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(p=.02, inplace=True),
        nn.Linear(in_features=1280, out_features=3, bias=True)
    )

    return model
    

# %%
effnetb0 = create_effnetb0()

summary(model=effnetb0, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )

# %%


# %%
from modules.engine import train_step, test_step
from tqdm import tqdm


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter) -> dict[str, list]:
    
    results = {
        "train_loss" : [],
        "train_acc" : [],
        "test_loss" : [],
        "test_acc" : []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model,
                                           train_dataloader,
                                           loss_fn,
                                           optimizer,
                                           device)
        
        print(f"epoch {epoch+1} / {epochs}")
        print(f"Train loss: {train_loss}")
        print(f"Train acc: {train_acc}")
        

        test_loss, test_acc = test_step(model,
                                    test_dataloader,
                                    loss_fn,
                                    device)
        print(f"Test loss: {test_loss}")
        print(f"Test acc: {test_acc}")
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)


        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)

        if writer:
            writer.add_scalars(main_tag="Loss", 
                            tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                            global_step=epoch)

            # Add accuracy results to SummaryWriter
            writer.add_scalars(main_tag="Accuracy", 
                            tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc}, 
                            global_step=epoch)
            
            # Track the PyTorch model architecture
            writer.add_graph(model=model, 
                            # Pass in an example input
                            input_to_model=torch.randn(32, 3, 224, 224).to(device))
        
            writer.close()
        
        

    return results
def set_seeds(seed:int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def create_writer(experiment_name:str,
                  model_name:str,
                  extra: str = None):
    from datetime import datetime
    import os 

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    return SummaryWriter(log_dir=log_dir)

# %%
from modules.utils import save_model

set_seeds()

num_epochs = [5, 10]

models = ["effnetb0", "effnetb2"]

train_10_b0, test_10_b0, train_20_b0, test_20_b0 = create_dataloaders(224)

train_10_b2, test_10_b2, train_20_b2, test_20_b2 = create_dataloaders(260)

train_dataloaders = {"data_10_percent" : {"effnetb0" : [train_10_b0, test_10_b0], "effnetb2" : [train_10_b2, test_10_b2]},
                     "data_20_percent" : {"effnetb0" : [train_20_b0, test_20_b0], "effnetb2" : [train_20_b2, test_20_b2]}}

experiment_number = 0

for dataloader_name in train_dataloaders:

    data_perecent = train_dataloaders[dataloader_name]


    for epochs in num_epochs:

        for model_name in models:

            train_dataloader, test_dataloader = data_perecent[model_name]


            experiment_number += 1
            print(f"[INFO] Experiment number: {experiment_number}")
            print(f"[INFO] Model: {model_name}")
            print(f"[INFO] DataLoader: {dataloader_name}")
            print(f"[INFO] Number of epochs: {epochs}")

            if (model_name == "effnetb0"):
                model = create_effnetb0().to(device)
            elif (model_name == "effnetb2"):
                model = create_effnetb2().to(device)

            train(model=model,
                  train_dataloader=train_dataloader,
                  test_dataloader=test_dataloader,
                  optimizer= torch.optim.Adam(model.parameters(), lr=0.001),
                  loss_fn= nn.CrossEntropyLoss(),
                  epochs=epochs,
                  device=device,
                  writer=create_writer(experiment_name=dataloader_name,
                                       model_name=model_name,
                                       extra=f"{epochs}_epochs")
            )

            save_filepath = f"07_{model_name}_{dataloader_name}_{epochs}_epochs.pth"
            save_model(model=model,
                       target_dir="models",
                       model_name=save_filepath)
            print("-"*50 + "\n")
        



