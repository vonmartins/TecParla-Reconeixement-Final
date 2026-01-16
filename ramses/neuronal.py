import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ramses.mod import Modelo
from ramses.util import *

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Sigmoid()
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

class RedNeuronal(Modelo):
    def __init__(self, pathMod=None, lisMod=None, hidden_sizes=[64, 32], activation='relu', epochs=50):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.epochs = epochs
        
        if pathMod != None:
            self.leeMod(pathMod)
        elif lisMod != None:
            self.unidades = leeLis(lisMod)
            self.unidad_to_idx = {u: i for i, u in enumerate(self.unidades)}
            self.idx_to_unidad = {i: u for i, u in enumerate(self.unidades)}
        else:
            raise("Hay que indicar el fichero del modelo o la lista de unidades")
    
    def leeMod(self, pathMod):
        datos = torch.load(pathMod)
        self.unidades = datos['unidades']
        self.unidad_to_idx = datos['unidad_to_idx']
        self.idx_to_unidad = datos['idx_to_unidad']
        self.hidden_sizes = datos['hidden_sizes']
        self.activation = datos['activation']
        
        input_size = datos['input_size']
        output_size = len(self.unidades)
        
        self.model = MLP(input_size, self.hidden_sizes, output_size, self.activation)
        self.model.load_state_dict(datos['model_state'])
        self.model.eval()

    def escMod(self, pathMod):
        chkPathName(pathMod)
        torch.save({
            'unidades': self.unidades,
            'unidad_to_idx': self.unidad_to_idx,
            'idx_to_unidad': self.idx_to_unidad,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'input_size': self.input_size,
            'model_state': self.model.state_dict()
        }, pathMod)

    def inicMod(self):
        self.X_train = []
        self.y_train = []

    def __add__(self, prm_unidad):
        prm, unidad = prm_unidad
        self.X_train.append(prm)
        self.y_train.append(self.unidad_to_idx[unidad])
        return self
    
    def calcMod(self):
        X = torch.FloatTensor(np.array(self.X_train))
        y = torch.LongTensor(np.array(self.y_train))
        
        self.input_size = X.shape[1]
        output_size = len(self.unidades)
        
        self.model = MLP(self.input_size, self.hidden_sizes, output_size, self.activation)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        batch_size = 64
        n_samples = X.shape[0]
        
        for epoch in range(self.epochs):
            indices = torch.randperm(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        self.model.eval()

    def __call__(self, prm):
        with torch.no_grad():
            X = torch.FloatTensor(prm).unsqueeze(0)
            output = self.model(X)
            _, predicted = torch.max(output, 1)
            return self.idx_to_unidad[predicted.item()]
