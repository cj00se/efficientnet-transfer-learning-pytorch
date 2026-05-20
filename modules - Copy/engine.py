import torch
from tqdm.auto import tqdm
import torch.utils


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):

    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X,y) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 
        loss.backward()
        optimizer.step()

        y_pred_class = y_pred.softmax(dim=1).argmax(dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device):
    
    model.eval()
    with torch.inference_mode():
        test_loss, test_acc = 0, 0
        for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing"):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y__pred_class = y_pred.softmax(dim=1).argmax(dim=1)
            test_acc += (y__pred_class == y).sum().item() / len(y_pred)
        
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> dict[str, list]:
    
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
        utils.save_model(model=model,
                    target_dir="models",
                    model_name="05_going_modular_script_mode.pth")
        

    return results