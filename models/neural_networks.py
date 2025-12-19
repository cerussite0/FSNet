from torch import nn

# Neural Network Models
# Multi-Layer Perceptron (MLP) with adjustable dropout and layer count  

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0, activation=nn.SiLU()):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), activation]

        for i in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout/(i+1))) # Adjust dropout rate for each layer
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid()) # Sigmoid activation for output layer
        self.mlp = nn.Sequential(*layers)
        # initialize weights if necessary               
    def forward(self, x):
        return self.mlp(x)
    


class Bilinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.l1 = nn.Linear(in_features/2, out_features, bias=bias)
        self.l2 = nn.Linear(in_features/2, out_features, bias=bias)
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        y1 = self.l1(x1)
        y2 = self.l2(x2)
        return y1 * y2
    
class BilinearMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super(BilinearMLP, self).__init__()
        layers = [Bilinear(input_dim, hidden_dim)]

        for i in range(num_layers - 1):
            layers += [Bilinear(hidden_dim, hidden_dim)]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout/(i+1))) # Adjust dropout rate for each layer

        layers.append(Bilinear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)
        # initialize weights if necessary
    def forward(self, x):
        return self.mlp(x)
    