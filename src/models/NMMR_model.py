import torch
import torch.nn as nn

#####ANN used in NMMR########
##########adopted from https://github.com/beamlab-hsph/Neural-Moment-Matching-Regression/blob/main/src/models/NMMR/NMMR_model.py #####


class MLP_for_NMMR(nn.Module):
    
    def __init__(self, input_dim, train_params):
        super(MLP_for_NMMR, self).__init__()

        self.train_params = train_params
        self.network_width = train_params["network_width"]
        self.network_depth = train_params["network_depth"]

        self.layer_list = nn.ModuleList()
        for i in range(self.network_depth):
            if i == 0:
                self.layer_list.append(nn.Linear(input_dim, self.network_width))
            else:
                self.layer_list.append(nn.Linear(self.network_width, self.network_width))
        self.layer_list.append(nn.Linear(self.network_width, 1))

    def forward(self, x):
        for ix, layer in enumerate(self.layer_list):
            if ix == (self.network_depth + 1):  # if last layer, don't apply relu activation
                x = layer(x)
            else:
                x = torch.relu(layer(x))

        return x
