import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

class AtomicDataset(Dataset):
    def __init__(self, data):
        """
        data_df - dataframe with all data [feature1, ..., featureN, BasisAtoms, E]
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        features = self.data.iloc[item].values[:-2]
        basis_atoms = self.data.iloc[item].values[-2]
        target = self.data.iloc[item].values[-1]
        return (
            torch.tensor(features.astype(float)),
            torch.tensor(target),
            torch.tensor(basis_atoms / 1),
        )


def make_dataframe(structures, basis_atoms, target):
    """
    make_dataframe create dataframe consisting of all
    structures suitable for AtomicDataset

    structures: array of features for different structures
    base_atoms: array of number of base atoms in each structure
    target: array target values for each structure
    """

    assert (
        len(structures) == len(basis_atoms) == len(target)
    ), "all passed lengths should be the same"

    combined_df = pd.DataFrame(columns=structures[0].columns.values)
    combined_df["BasisAtoms"] = ""
    combined_df["Target"] = ""
    for _struct, _basis, _target in zip(structures, basis_atoms, target):

        new_df = _struct.copy()
        new_df["BasisAtoms"] = _basis
        new_df["Target"] = _target

        combined_df = pd.concat([combined_df, new_df])

    return combined_df
