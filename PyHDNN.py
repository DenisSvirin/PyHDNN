import torch.nn as nn

class PyHDNN(nn.Module):
  """
  This neural network represents networks for all kind of
  atoms in the system.
  (currently 1)
  """
  def __init__(self, input_shape):
    super().__init__()
    self.atom_network = nn.Sequential(
        nn.Linear(input_shape, 120),
        nn.ELU(),
        nn.Linear(120, 120),
        nn.ELU(),
        nn.Linear(120, 1)
    )

  def forward(self, X):
    out = self.atom_network(X)
    return out
