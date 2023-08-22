import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from Dataset import AtomicDataset, make_dataframe
from Atomic_Descriptors import SymmetryFunctions
from PyHDNN import PyHDNN
from torch.utils.data import DataLoader
import torch.nn as nn

def train_epoch(model, optimizer, criterion):
    train_loss_log = []

    model.train()
    for x_batch, y_batch, basis in train_loader:
        x_batch, y_batch, basis = (
            x_batch.to(device),
            y_batch.to(device),
            basis.to(device),
        )

        optimizer.zero_grad()
        output = model(x_batch)

        loss = criterion(output * basis.reshape(-1, 1), y_batch.reshape(-1, 1))
        loss.backward()
        optimizer.step()
        train_loss_log.append(loss.item())
    return train_loss_log

@torch.no_grad()
def test_epoch(model, criterion):
    test_loss_log = []

    model.eval()
    for x_batch, y_batch, basis in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        output = model(x_batch)

        loss = criterion(output / basis, y_batch)

        test_loss_log.appned(loss)
    return test_loss_log


def train(model, optimizer, criterion, epochs):
    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_log_loss = train_epoch(model, optimizer, criterion)
        # test_log_loss = test_epoch(model, criterion)
    return np.mean(train_log_loss)


if __name__ == "__main__":
    data_coord = pd.read_csv("/Users/wexumin/Documents/HNN/data/fcc_coords.txt")
    dimer_data = pd.read_csv("/Users/wexumin/Documents/HNN/data/dimer_r_E.csv")
    fcc_data = pd.read_csv("/Users/wexumin/Documents/HNN/data/fcc_r_E.csv")

    dimer_r = dimer_data['r'].values
    dimer_E = dimer_data['E'].values

    x = data_coord['x'].values.reshape(-1,1)
    y = data_coord['y'].values.reshape(-1,1)
    z = data_coord['z'].values.reshape(-1,1)
    relative_coordinates = np.hstack((x,y,z))

    fcc_r = fcc_data['r'].values
    fcc_E = fcc_data['E'].values

    sf_fcc = SymmetryFunctions(
        len_g1_functions=5,
        len_g2_functions=21,
        lattice_constants=fcc_r,
        atomic_coordinates=relative_coordinates,
    )
    sf_fcc.transform_to_SymmetryFunctions()

    sf_dimer = SymmetryFunctions(
        len_g1_functions=5,
        len_g2_functions=21,
        lattice_constants=dimer_r[:-8],
        atomic_coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        no_angles=True,
    )
    sf_dimer.transform_to_SymmetryFunctions()

    dataset = AtomicDataset(
        make_dataframe(
            [sf_fcc.symmetry_functions_dataframe, sf_dimer.symmetry_functions_dataframe],
            [4, 2],
            [fcc_E, dimer_E[:-8]],
        )
    )
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = PyHDNN(52).double()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    loss = train(model, optimizer, criterion, 750)
    print("loss: ", loss)
