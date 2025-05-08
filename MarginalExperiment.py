
import numpy as np
from scipy.stats import norm, multivariate_normal, multivariate_t, t, uniform, expon, laplace, lognorm
import torch
#from statsmodels.graphics.gofplots import qqplot
from my_classes.NeuralCopula import MarginalModel, CopulaModel, NeuralCopula
from matplotlib import pyplot as plt

## Create dictionary with datasets from different distributions
datasets = {
    'Gaussian': np.random.normal(size=10000),
    'Student-t': np.random.standard_t(df=5, size=10000),
    'Uniform': np.random.uniform(size=10000),
    'Exponential': np.random.exponential(size=10000),
    'Laplace': np.random.laplace(size=10000),
    'LogNormal': np.random.lognormal(size=10000),
}


scaling = 1.0
modelDict = {}
for name, data in datasets.items():
    #Normalize the data
    print(f"Training model for {name} distribution")
    model = MarginalModel(num_layers=6, num_neurons=10, lr=0.01)
    data_tensor = model.CreateNormalizedTensor(data, scaling=scaling)
    # Train the model on the dataset
    model.train_model(data_tensor, epochs=5000, log_interval=500)
    # model.PlotModel()
    modelDict[name] = model




# Distributions and their parameters
distributions = {
    'Gaussian': (norm, {'loc': 0, 'scale': 1}),
    'Student-t': (t, {'df': 5}),
    'Uniform': (uniform, {'loc': 0, 'scale': 1}),
    'Exponential': (expon, {'scale': 1}),
    'Laplace': (laplace, {'loc': 0, 'scale': 1}),
    'LogNormal': (lognorm, {'s': 1.0, 'loc': 0.0, 'scale': np.exp(0.0)}),
}

for name, data in datasets.items(): 
    print(f"Validating neural copula against true model for {name} distribution")
    ## Data as tensor
    #data_tensor = torch.tensor(data, dtype=torch.float32).view(-1, 1)
    data_tensor = model.CreateNormalizedTensor(data, scaling=scaling)[:-2,:]  # Exclude the last two points
    U_NeuralNet = modelDict[name].evaluateCDFData(data_tensor)

    dist = distributions[name][0]
    params = distributions[name][1]
    dist_obj = dist(**params)
    U_True = dist_obj.cdf(data)

    plt.figure(figsize=(6, 6))
    plt.scatter(U_NeuralNet, U_True, alpha=0.5, label=f"{name} data", s=12)
    plt.xlabel("U_NeuralNet")
    plt.ylabel("U_True")
    plt.title(f"QQ-Plot of {name} data")
    plt.plot([0, 1], [0, 1], 'r--', label='y=x')
    plt.show()

    U_NeuralNet = U_NeuralNet.flatten()
    mse = np.mean(np.abs(U_NeuralNet - U_True))
    print(f"Mean Squared Error for {name} distribution: {mse:.4f}")
    meanError = np.mean(U_NeuralNet - U_True)
    print(f"Mean Error for {name} distribution: {meanError:.4f}")
    print(np.count_nonzero(U_NeuralNet < U_True))
    print('------------------------------------------------------------')