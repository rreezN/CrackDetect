import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from hydra import Hydra, SparseScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeCV
from src.data.dataloader import Platoon
from src.hydra.hydra_regressor import Regressor


# load data (torch.FloatTensor, shape = (num_examples, 1, length))
def load_data():
    gm_segment = pd.read_csv('data/processed/gm/segment_000.csv', sep=';', encoding='utf8', engine='pyarrow')
    aran_segment = pd.read_csv('data/processed/aran/segment_000.csv', sep=';', encoding='utf8', engine='pyarrow').fillna(0)

    acc_z = gm_segment['acc.xyz_2'].to_numpy()
    X_train = acc_z[:-500]
    X_test = acc_z[-500:]
    
    KPIS = calculate_kpis(aran_segment)
    y_train = KPIS[:-500]
    y_test = KPIS[-500:]
    
    

    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train).view(1, 1, -1)
    X_test = torch.FloatTensor(X_test).view(1, 1, -1)
    y_train = torch.FloatTensor(y_train).view(1, -1)
    y_test = torch.FloatTensor(y_test).view(1, -1)
    
    # Simulate batches
    repeats = 10
    X_train = X_train.repeat(repeats, 1, 1)
    X_test = X_test.repeat(repeats, 1, 1)
    y_train = y_train.repeat(repeats, 1)
    y_test = y_test.repeat(repeats, 1)
    
    # Add random noise cause why not
    X_train += torch.randn_like(X_train) * 0.1
    X_test += torch.randn_like(X_test) * 0.1
    y_train += torch.randn_like(y_train) * 0.1
    y_test += torch.randn_like(y_test) * 0.1
    
    return X_train, y_train, X_test, y_test
    
    
    
def calculate_kpis(aran_segment):
    IRI = iri_mean(aran_segment)
    return IRI


def iri_mean(aran_segment):
    IRL = aran_segment['Venstre IRI (m/km)']
    IRR = aran_segment['Højre IRI (m/km)']
    return (((IRL + IRR) / 2)**(0.2)).to_numpy()


if __name__ == '__main__':
    # load data
    X_train, y_train, X_test, y_test = load_data()

    transform = Hydra(X_train.shape[-1])

    X_training_transform = transform(X_train)
    X_test_transform = transform(X_test)

    scaler = SparseScaler()

    X_training_transform = scaler.fit_transform(X_training_transform)
    X_test_transform = scaler.transform(X_test_transform)

    # Train model
    model = Regressor(X_training_transform.shape[-1], 1)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    criterion = torch.nn.MSELoss()
    for epoch in range(100):
        optimizer.zero_grad()
        for batch in range(0, X_training_transform.shape[0], 32):
            X = X_training_transform[batch:batch + 32]
            y = y_train[batch:batch + 32]
            predictions = model(X)
            loss = criterion(predictions, y.mean(1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    model.eval()
    
    predictions = model(X_test_transform)
    
    for i in range(predictions.shape[0]):
        print(f'Prediction: {predictions[i].item()}, True: {y_test[i].mean().item()}')

        
    