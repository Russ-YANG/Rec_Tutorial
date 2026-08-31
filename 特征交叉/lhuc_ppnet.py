import torch
import torch.nn as nn
import torch.nn.functional as F

class LHUC_PPNet(nn.Module):

    def __init__(self, 
                 user_dim, 
                 item_dim, 
                 hidden_dims=[128, 64], 
                 dropout_rate=0.2):

        super(LHUC_PPNet, self).__init__()
        
        self.main_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        input_dim = user_dim + item_dim

        for hidden_dim in hidden_dims:
            self.main_layers.append(nn.Linear(input_dim, hidden_dim))
            self.gate_layers.append(nn.Linear(user_dim, hidden_dim))
            input_dim = hidden_dim

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_dims = hidden_dims
    
    def forward(self, user_features, item_features):

        representation = torch.cat([user_features, item_features], dim=1)

        for i in range(len(self.hidden_dims)):
            representation = self.main_layers[i](representation)
            representation = F.relu(representation)

            gate = self.gate_layers[i](user_features)
            gate = 2.0 * torch.sigmoid(gate)
            representation = representation * gate
            representation = self.dropout(representation)

        output = self.output_layer(representation)
        
        return output.squeeze(1)

