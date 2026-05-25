import os
from os.path import join

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import binned_statistic

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from swag import data, models, utils, losses
from swag.posteriors import SWAG

# Load the diabetes dataset
diabetes_X, diabetes_y  = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:, np.newaxis, 2]
criterion = torch.nn.MSELoss()

print("load data")

def compute_loss(model, x, y):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    # neg_log_likelihood = -y_hat.log_prob(y)
    # loss = torch.mean(neg_log_likelihood)
    return loss

def train(model, optimizer, x_train, y_train, n_epochs, batch_size, scheduler=None, print_every=10):
    train_losses, val_losses = [], []
    for epoch in range(n_epochs):
        batch_indices = sample_batch_indices(x_train, y_train, batch_size)
        
        batch_losses_t = []
        for batch_ix in batch_indices:
            model.train()
            optimizer.zero_grad()
            
            loss = compute_loss(model, x_train[batch_ix], y_train[batch_ix])
            loss.backward()
            optimizer.step()
            # model.eval()
            # batch_losses_t.append(b_train_loss.detach().numpy())
            
        # if scheduler is not None:
        #     scheduler.step()
            
        # train_loss = np.mean(batch_losses_t)
        # train_losses.append(train_loss)
    # return train_losses, val_losses


def sample_batch_indices(x, y, batch_size, rs=None):
    if rs is None:
        rs = np.random.RandomState()
    
    train_ix = np.arange(len(x))
    rs.shuffle(train_ix)
    
    n_batches = int(np.ceil(len(x) / batch_size))
    
    batch_indices = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch_indices.append(
            train_ix[start:end].tolist()
        )

    return batch_indices

class DeepNormalModel(torch.nn.Module):
    def __init__(
        self, 
        n_inputs,
    ):
        super().__init__()
        
        self.jitter = 1e-6
        self.swish = torch.nn.SiLU()
        self.s1 = torch.nn.Linear(n_inputs, 8)  
        self.s2 = torch.nn.Linear(8, 8)  
        self.out = torch.nn.Linear(8, 1)   
        
    def forward(self, x):
        x = self.s1(x)
        x = self.swish(x)
        x = self.s2(x)
        x = self.out(x)
        return x
        mean = x[:, 0]
        std = F.softplus(x[:, 1]) + self.jitter
        return torch.distributions.Normal(mean, std)

learning_rate = 1e-2
momentum = 0.9
weight_decay = 1e-5

n_epochs = 100
batch_size = 32

swag_model = SWAG(DeepNormalModel, 
                subspace_type="covariance", subspace_kwargs={'max_rank': 10},
                n_inputs = diabetes_X.shape[1])

optimizer = torch.optim.Adam(
    swag_model.parameters(), 
    lr=learning_rate, 
    # momentum  = 0.9,
    weight_decay = 1e-4,
)

diabetes_X = torch.from_numpy(diabetes_X.astype(np.float32))
diabetes_y = torch.from_numpy(diabetes_y.astype(np.float32))

train(
    swag_model, 
    optimizer, 
    diabetes_X, 
    diabetes_y, 
    n_epochs=n_epochs, 
    batch_size=batch_size, 
    print_every=10,
)

K = 100
for k in range(K):
    # samples = swag_model.sample()
    # swag_model.set_swa()
    y_hats = swag_model(diabetes_X)
    print(y_hats.mean())
    # diabetes_y_pred = y_hats.mean.detach().numpy()  
    # print(diabetes_y_pred.mean())
    # std = y_hats.stddev   

# The coefficients
print("Coefficients: \n", list(swag_model.parameters()))
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y, diabetes_y_pred))
print("NLLs: %.2f" % compute_loss(swag_model, diabetes_X, diabetes_y))

# Plot outputs
plt.scatter(diabetes_X, diabetes_y, color="black")
plt.plot(diabetes_X, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
plt.savefig('books_read.png')