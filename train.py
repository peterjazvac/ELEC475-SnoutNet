import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
import time
from model import SnoutNet
from datasetLoader import train_loader, test_loader

# Define the EuclideanDistanceLoss class
class EuclideanDistanceLoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(torch.sum((pred - target) ** 2, dim=1)).mean()

def train(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, scheduler, device, save_file=None, plot_file=None):
    print('training ...')
    model.train()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience, trials = 5, 0  # Early stopping parameters

    for epoch in range(1, n_epochs+1):
        start_time = time.time()  # Start time for the epoch
        print('epoch ', epoch)
        loss_train = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            print(f'Processing batch {batch_idx + 1}/{len(train_loader)}')
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step(loss_train)
        train_losses.append(loss_train / len(train_loader))

        # Validation
        model.eval()
        loss_val = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss_val += loss.item()
        val_losses.append(loss_val / len(val_loader))

        epoch_time = time.time() - start_time  # Calculate epoch duration
        print('{} Epoch {}, Training loss {}, Validation loss {}, Time taken: {:.2f} seconds'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader), loss_val / len(val_loader), epoch_time))

        # Early stopping
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            trials = 0
            if save_file is not None:
                torch.save(model.state_dict(), save_file)
        else:
            trials += 1
            if trials >= patience:
                print(f"Early stopping on epoch {epoch}")
                break

        if plot_file is not None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()
            plt.plot(train_losses, label='train')
            plt.plot(val_losses, label='val')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            print('saving ', plot_file)
            plt.savefig(plot_file)

def main():
    save_file = 'weights.pth'
    n_epochs = 50  # Increase the number of epochs
    plot_file = 'plot.png'

    print('running main ...')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\t\tusing device ', device)

    model = SnoutNet()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_fn = EuclideanDistanceLoss()

    train(
        n_epochs=n_epochs,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=test_loader,
        scheduler=scheduler,
        device=device,
        save_file=save_file,
        plot_file=plot_file
    )

if __name__ == '__main__':
    main()