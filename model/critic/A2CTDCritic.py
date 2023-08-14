import torch.nn.functional as F
import torch.nn as nn
import torch

from utils import get_regularization

import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim = 1, dropout_rate = 0., do_batch_norm = True):
        super(DNN, self).__init__()
        self.in_dim = in_dim
        layers = []

        # hidden layers
        for hidden_dim in hidden_dims:
            linear_layer = nn.Linear(in_dim, hidden_dim)
            # torch.nn.init.xavier_uniform_(linear_layer.weight, gain=nn.init.calculate_gain('relu'))
            layers.append(linear_layer)
            in_dim = hidden_dim

            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            if do_batch_norm:
#                 layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LayerNorm([hidden_dim]))

        # prediction layer
        last_layer = nn.Linear(in_dim, out_dim)
        layers.append(last_layer)
        # torch.nn.init.xavier_uniform_(last_layer.weight, gain=1.0)

        self.layers = nn.Sequential(*layers)
        
    def forward(self, inputs):
        """
        @input:
            `inputs`, [bsz, in_dim]
        @output:
            `logit`, [bsz, out_dim]
        """
#         inputs = inputs.view(-1, self.in_dim)
        logit = self.layers(inputs)
        return logit

class A2CTDCritic(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_hidden_dims
        - critic_dropout_rate
        '''
        parser.add_argument('--critic_hidden_dims', type=int, nargs='+', default=[128], 
                            help='specificy a list of k for top-k performance')
        parser.add_argument('--critic_dropout_rate', type=float, default=0.2, 
                            help='dropout rate in deep layers')
        return parser
    
    def __init__(self, args, environment, policy):
        super().__init__()
        self.state_dim = policy.state_dim
        
        self.net = DNN(self.state_dim, args.critic_hidden_dims, 1, 
                       dropout_rate = args.critic_dropout_rate, do_batch_norm = True)
        
    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {'state_emb': (B, state_dim), 'action_emb': (B, action_dim)}
        '''
        
#         state_emb = self.state_encoder(feed_dict)['state_emb'].view(-1, self.state_dim)
#         import pdb
#         pdb.set_trace()
        state_emb = feed_dict['state_emb']
        
        
#         import pdb
#         pdb.set_trace()
        
        V = self.net(state_emb).squeeze()
        reg = get_regularization(self.net)
        return {'reg': reg, 'v':V}
