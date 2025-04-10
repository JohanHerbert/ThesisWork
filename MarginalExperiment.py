import numpy as np
from scipy.stats import norm, multivariate_normal, multivariate_t, t
import torch
from statsmodels.graphics.gofplots import qqplot
from my_classes.NeuralCopula import MarginalModel, CopulaModel, NeuralCopula

## Creating data
np.random.seed(0)
torch.manual_seed(0)
## Generate sample
Z = np.random.standard_normal(10000)
scaling = 3.0
boundaryPoints = scaling * np.array([np.max(Z), np.min(Z)])

Z = np.concatenate((Z, boundaryPoints))
Z_squashed = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))

X = torch.tensor(Z_squashed, dtype=torch.float32).view(-1, 1)

# Initialize and train the model with custom layers and neurons
model = MarginalModel(num_layers=6, num_neurons=10, lr=0.01)
model.train_model(X, epochs=5000, log_interval=500)






