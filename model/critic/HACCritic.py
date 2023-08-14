import torch.nn.functional as F
import torch.nn as nn
import torch

from model.components import DNN
from utils import get_regularization

class HACCritic(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_hidden_dims
        - critic_dropout_rate
        '''
        parser.add_argument('--critic_hidden_dims', type=int, nargs='+', default=[256], 
                            help='specificy a list of k for top-k performance')
        parser.add_argument('--critic_dropout_rate', type=float, default=0.2, 
                            help='dropout rate in deep layers')
        parser.add_argument('--critic_slate_size', type=int, default=6, 
                            help='list size for rl')
        return parser
    
    def __init__(self, args, environment, policy):
        super().__init__()
        self.state_dim = policy.state_dim
        self.ls_size = args.critic_slate_size
        self.f_dim = policy.state_encoder.f_dim
#         self.state_encoder = policy.state_encoder
        self.net = DNN(self.state_dim + self.f_dim*self.ls_size, args.critic_hidden_dims, 1, 
                       dropout_rate = args.critic_dropout_rate, do_batch_norm = True)
        
    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {'state_emb': (B, state_dim), 'action_emb': (B, action_dim)}
        '''
        
        state_emb = feed_dict['state_emb']
#         state_emb = self.state_encoder(feed_dict)['state_emb'].view(-1, self.state_dim)
        ls_emb_ = feed_dict['list_emb']
        ls_emb = torch.flatten(ls_emb_, start_dim=1)
        
        Q = self.net(torch.cat((state_emb, ls_emb), dim = -1)).view(-1)
#         reg = get_regularization(self.net, self.state_encoder)
        
        reg = get_regularization(self.net)
        return {'q': Q, 'reg': reg}
