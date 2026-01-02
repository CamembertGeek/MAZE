import matplotlib.pyplot as plt
import torch
from dataloader import DatasetBuilder
from U_net import Unet





if __name__ == "__main__":

    training_file_path = "/home/louis/Documents/Programs/365/MAZE/Training/npy_files/training_grid.pt"
    training_solved_file_path = "/home/louis/Documents/Programs/365/MAZE/Training/npy_files/solved_grid.pt"

    # Creation of the dataset
    Dataloader = DatasetBuilder(training_file_path=training_file_path, solved_file_path=training_solved_file_path)
    train_loader, val_loader = Dataloader.get_dataloaders(batch_size=16, num_workers=2, pin_memory=True)

    # Some verifocation
    # x, y = next(iter(train_loader))
    # print(x.shape)
    # print(y.shape)

    # print(x.min(), x.max())
    # print(y.min(), y.max())
    # print(x[:,1].sum(dim=(1,2)).unique())
    # print(x[:,2].sum(dim=(1,2)).unique())

    # print(len(train_loader.dataset))
    # print(len(val_loader.dataset))
    # print(len(train_loader))

    # Trainig of the model
    unet = Unet(n_input=3, n_output=1, loss='BCEWithLogitsLoss', activation_fcn='ReLU', learning_rate=3e-4, device='cuda')
    print(next(unet.parameters()).device)
    print(torch.cuda.is_available())
    history = unet.train_model(train_loader=train_loader, val_loader=val_loader, epochs=2, grad_clip=None)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    # Plot of the training and the validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, color='r', label="Training loss")
    plt.plot(epochs, val_loss, color='b', label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
