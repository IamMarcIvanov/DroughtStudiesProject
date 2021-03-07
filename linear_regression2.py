import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F


class R2Loss(nn.Module):
    def forward(self, y_pred, y):
        var_y = torch.var(y, unbiased=False)
        return 1.0 - F.mse_loss(y_pred, y, reduction="mean") / var_y


def normalize(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / std


def training_loop(nos_epochs, x_train, x_test, y_train, y_test):

    model = nn.Linear(9, 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_mse = nn.MSELoss()
    loss_r2 = R2Loss()

    train_loss_list_MSE, test_loss_list_MSE = [], []
    train_loss_list_R2, test_loss_list_R2 = [], []

    for epoch in range(1, nos_epochs + 1):
        y_train_pred = model(x_train)
        train_loss_MSE = loss_mse(y_train_pred, y_train)
        train_loss_R2 = loss_r2(y_train_pred, y_train)
        train_loss_list_MSE.append(train_loss_MSE.item())
        train_loss_list_R2.append(train_loss_R2.item())

        y_test_pred = model(x_test)
        test_loss_MSE = loss_mse(y_test_pred, y_test)
        test_loss_R2 = loss_r2(y_test_pred, y_test)
        test_loss_list_MSE.append(test_loss_MSE.item())
        test_loss_list_R2.append(test_loss_R2.item())

        optimizer.zero_grad()
        train_loss_MSE.backward()
        optimizer.step()
    
    print("{:<10} {:<10} {:<15}".format('Loss Type', 'Epoch', 'Loss'))
    plot_and_print_loss(nos_epochs, train_loss_list_MSE, test_loss_list_MSE, "MSE")
    plot_and_print_loss(nos_epochs, train_loss_list_R2, test_loss_list_R2, "R2")


def plot_and_print_loss(nos_epochs, train_loss, test_loss, loss_type):
    """below is printing loss every 50th item"""
    # s = "=" * 72 + "\n\n"
    # print(
        # s
        # + "{:^10} {:<30} {:<30}".format(
            # "Epoch", loss_type + " Train Loss", loss_type + " Test Loss"
        # )
    # )
    # for loss1, loss2 in zip(train_loss[::50], test_loss[::50]):
        # ind = train_loss.index(loss1)
        # print("{:^10} {:<30} {:<30}".format(ind + 1, loss1, loss2))
    """below is plotting loss"""
    # epoch_list = list(range(1, nos_epochs + 1))
    # plt.plot(epoch_list, train_loss, "r")
    # plt.plot(epoch_list, test_loss, "b")
    # plt.xlabel("epochs")
    # plt.ylabel("losses")
    # plt.legend(["train loss", "test loss"])
    # plt.title(loss_type + " Loss")
    # plt.show()
    """below is printing min loss and epoch"""
    train_loss_abs = [abs(x) for x in train_loss]
    test_loss_abs = [abs(x) for x in test_loss]
    train = min(train_loss_abs)
    test = min(test_loss_abs)
    train_ind = train_loss_abs.index(train)
    test_ind = test_loss_abs.index(test)
    print("{:<10} {:<10} {:<15}".format(loss_type + "Train", train_ind, train_loss[train_ind]))
    print("{:<10} {:<10} {:<15}".format(loss_type + "Test", test_ind, test_loss[test_ind]))


rain_path = r"E:\BITS\Yr 3 Sem 2\CE F376 Civil Climate change SOP\Drought Studies\Data\Hyderabad - Hyd_Rainfall.csv"
rain = pd.read_csv(rain_path)
rain_tensor = torch.from_numpy(rain.values).float()


all_var_path = r"E:\BITS\Yr 3 Sem 2\CE F376 Civil Climate change SOP\Drought Studies\Data\Hyderabad - Hyderabad_data - Copy.csv"
all_var = pd.read_csv(all_var_path)
all_var_tensor = torch.from_numpy(all_var.values).float()


nos_samples = all_var_tensor.shape[0]
nos_validation_samples = int(0.2 * nos_samples)
shuffled_indices = torch.randperm(nos_samples)
train_indices = shuffled_indices[:-nos_validation_samples]
validation_indices = shuffled_indices[-nos_validation_samples:]

x_train = normalize(all_var_tensor[train_indices])
x_test = normalize(all_var_tensor[validation_indices])

y_train = rain_tensor[train_indices]
y_test = rain_tensor[validation_indices]

training_loop(
    nos_epochs=5000,
    x_train=x_train,
    x_test=x_test,
    y_train=y_train,
    y_test=y_test,
)
