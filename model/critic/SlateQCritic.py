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

class BaseStateEncoder(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - state_encoder_feature_dim
        - state_encoder_attn_n_head
        - state_encoder_hidden_dims
        - state_encoder_dropout_rate
        '''
        parser.add_argument('--state_encoder_feature_dim', type=int, default=32, 
                            help='dimension size for state')
        parser.add_argument('--state_encoder_attn_n_head', type=int, default=4, 
                            help='dimension size for all features')
        parser.add_argument('--state_encoder_hidden_dims', type=int, nargs='+', default=[128], 
                            help='specificy a list of k for top-k performance')
        parser.add_argument('--state_encoder_dropout_rate', type=float, default=0.1, 
                            help='dropout rate in deep layers')
        return parser
    
    def __init__(self, args, environment):
        super().__init__()
        # action space
        self.item_dim = environment.action_space['item_feature'][1]
        # observation space
        user_profile_info = environment.observation_space['user_profile']
        
        self.user_dim = user_profile_info[1]
        # state space
        self.f_dim = args.state_encoder_feature_dim
        self.state_dim = self.f_dim # + self.user_dim
        # policy network modules
        self.user_profile_encoder = DNN(self.user_dim, args.state_encoder_hidden_dims, self.f_dim, 
                                        dropout_rate = args.state_encoder_dropout_rate, do_batch_norm = True)
        self.item_emb_layer = nn.Linear(self.item_dim, self.f_dim)
        self.seq_user_attn_layer = nn.MultiheadAttention(self.f_dim, args.state_encoder_attn_n_head, batch_first = True)
        self.state_linear = nn.Linear(self.f_dim + self.user_dim, self.state_dim)
        self.state_norm = nn.LayerNorm([self.state_dim])
        # To be implemented: action modules
    
    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {'user_profile': (B, user_dim), 
                    'history_features': (B, H, item_dim, 
                    'candidate_features': (B, L, item_dim) or (1, L, item_dim)}
        @model:
        - user_profile --> user_emb (B,1,f_dim)
        - history_items --> history_item_emb (B,H,f_dim)
        - (Q:user_emb, K&V:history_item_emb) --(multi-head attn)--> user_state (B,1,f_dim)
        - user_state --> action_prob (B,n_item)
        @output:
        - out_dict: {"action_emb": (B,action_dim), 
                     "state_emb": (B,f_dim),
                     "reg": scalar,
                     "action_prob": (B,L), include probability score when candidate_features are given}
        '''
        # user embedding (B,1,f_dim)
        user_emb = self.user_profile_encoder(feed_dict['user_profile']).view(-1,1,self.f_dim)
        B = user_emb.shape[0]
        # history embedding (B,H,f_dim)
        
        history_item_emb = self.item_emb_layer(feed_dict['history_features'])
        candidate_item_emb = self.item_emb_layer(feed_dict['candidate_features'])
        # cross attention, encoded history is (B,1,f_dim)
        user_state, attn_weight = self.seq_user_attn_layer(user_emb, history_item_emb, history_item_emb)
        # (B, 2*f_dim)
#         user_state = torch.cat((user_state.view(B, self.f_dim), user_emb.view(B, self.f_dim)), dim = -1)
#         user_state = torch.sigmoid(self.state_linear(user_state))
        user_state = self.state_linear(torch.cat((user_state.view(B, self.f_dim),
                                                  feed_dict['user_profile'].view(B,self.user_dim)), dim = -1))
        user_state = self.state_norm(user_state)
#         user_state = torch.sigmoid(user_state)
#         reg = get_regularization(self.user_profile_encoder, self.item_emb_layer, 
#                                  self.seq_user_attn_layer, self.action_layer)
        return {'state_emb': user_state,'item_emb': candidate_item_emb}


class SlateQCritic(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_hidden_dims
        - critic_dropout_rate
        '''
        parser = BaseStateEncoder.parse_model_args(parser)

        parser.add_argument('--critic_hidden_dims', type=int, nargs='+', default=[128], 
                            help='specificy a list of k for top-k performance')
        parser.add_argument('--critic_dropout_rate', type=float, default=0.2, 
                            help='dropout rate in deep layers')
        return parser
    
    def __init__(self, args, environment):
        super().__init__()

        self.dropout_rate = args.state_encoder_dropout_rate
        # action space
        self.f_dim = args.state_encoder_feature_dim
        self.item_space = environment.action_space['item_id'][1]
        self.item_dim = environment.action_space['item_feature'][1]
        # policy network modules
        self.state_encoder = BaseStateEncoder(args, environment)
        self.state_dim = self.state_encoder.state_dim

        self.action_dim = self.item_dim + 1
#         self.state_encoder = policy.state_encoder

        self.action_layer = DNN(self.state_dim, args.critic_hidden_dims, self.action_dim, 
                                dropout_rate = args.critic_dropout_rate, do_batch_norm = True)
        self.net = DNN(self.state_dim + self.f_dim , args.critic_hidden_dims, 1, 
                       dropout_rate = args.critic_dropout_rate, do_batch_norm = True)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {'state_emb': (B, state_dim), 'action_emb': (B, action_dim)}
        '''
        
        state_ = self.state_encoder(feed_dict)
        user_state = state_['state_emb']
        B = user_state.shape[0]
        state_emb = user_state.view(B,-1)
        
        state_emb_ = state_emb.unsqueeze(1)
        
        candi_fea = state_['item_emb']
        
        state_mod = state_emb_.repeat(1, candi_fea.shape[1],1) 
        # (B, L, item_dim)
        
        if candi_fea.shape[0] == 1:
            candi_mod = candi_fea.repeat(state_emb.shape[0],1,1) 
        else:
            candi_mod = candi_fea

#         import pdb
#         pdb.set_trace()
        # (B, L)
        Q = self.net(torch.cat((state_mod, candi_mod), dim = -1)).squeeze()
        
#         reg = get_regularization(self.net, self.state_encoder)
        reg = get_regularization(self.net)
        Q_reg = self.softmax(Q)
        return {'q_all': Q,'q_reg':Q_reg, 'reg': reg, 'state_emb': state_emb }
