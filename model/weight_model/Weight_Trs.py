import torch.nn.functional as F
import torch.nn as nn
import torch

from model.components import DNN
from utils import get_regularization


# def positional_encoding(max_seq_len, embed_dim):
#     position = torch.arange(0, max_seq_len).unsqueeze(1)
#     div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
#     pe = torch.zeros(max_seq_len, embed_dim)
#     pe[:, 0::2] = torch.sin(position * div_term)
#     pe[:, 1::2] = torch.cos(position * div_term)
#     return pe

# def get_pos_encoding(inputs):
#     max_seq_len, embed_dim = inputs.shape[-2],inputs.shape[-1]
#     position = torch.arange(0, max_seq_len).unsqueeze(1)
#     div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
#     pe = torch.zeros(max_seq_len, embed_dim)
#     pe[:, 0::2] = torch.sin(position * div_term)
#     pe[:, 1::2] = torch.cos(position * div_term)
#     return pos_encoding

def positional_encoding(x):
    batch_size, seq_length, embedding_dim = x.shape
    position = torch.arange(seq_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
    position_encoding = torch.zeros((seq_length, embedding_dim))
    position_encoding[:, 0::2] = torch.sin(position * div_term)
    position_encoding[:, 1::2] = torch.cos(position * div_term)
    position_encoding = position_encoding.unsqueeze(0).expand(batch_size, -1, -1)
    return position_encoding

class Weight_Trs(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_hidden_dims
        - critic_dropout_rate
        '''
        parser.add_argument('--critic_hidden_dims1', type=int, nargs='+', default=[256], 
                            help='specificy a list of k for top-k performance')
        parser.add_argument('--critic_dropout_rate1', type=float, default=0.2, 
                            help='dropout rate in deep layers')
        parser.add_argument('--critic_slate_size1', type=int, default=6, 
                            help='list size for rl')
        parser.add_argument('--embed_dim', type=int, default=256, 
                            help='embedding_dim')
        return parser
    
    def __init__(self, args, environment, policy):
        super().__init__()
        self.state_dim = policy.state_dim
        self.ls_size = args.critic_slate_size1
        self.f_dim = policy.state_encoder.f_dim
        self.embed_dim = args.embed_dim
#         self.state_encoder = policy.state_encoder

#         self.positional_encoding = get_pos_encoding(self.ls_size, self.embed_dim)

        self.embedding = nn.Linear(self.state_dim *2 + self.f_dim + 2, self.embed_dim)
        
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=8)
        self.net = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        self.fc = nn.Linear(self.embed_dim, 1, bias=True)

        
    def forward(self, curr_feed_dic, next_feed_dic, reward):
        '''
        @input:
        - feed_dict: {'state_emb': (B, state_dim), 'action_emb': (B, action_dim)}
        '''
        
        state_emb = curr_feed_dic['state_emb'].detach()
#         state_emb = self.state_encoder(feed_dict)['state_emb'].view(-1, self.state_dim)
        ls_emb = curr_feed_dic['list_emb'].detach()
        next_state_emb = next_feed_dic['state_emb'].detach()
        
        reward = reward.unsqueeze(dim=-1)
        state_emb = state_emb.unsqueeze(dim=1)
        state_emb = state_emb.repeat(1, 6, 1)
        next_state_emb = next_state_emb.unsqueeze(dim=1)
        next_state_emb = next_state_emb.repeat(1, 6, 1)
        
        reward_sum = reward.sum(dim=1,keepdim=True).repeat(1, 6, 1)
        
        x = torch.cat([state_emb, ls_emb, next_state_emb,reward,reward_sum], dim = 2)
        
        x = self.embedding(x)
        
        
        
        pos_x =positional_encoding(x).to(x.device)
        X = x + pos_x
        
        
        weight = self.fc(self.net(X)).squeeze()
        
        weight = F.softmax(weight, dim=1)
#         reg = get_regularization(self.net, self.state_encoder)
        
        reg = get_regularization(self.net)
        return {'weight': weight, 'reg': reg}
    
    
    
#     class TransformerModel(nn.Module):
#     def __init__(self, input_dim, embed_dim, max_seq_len):
#         super(TransformerModel, self).__init__()
#         self.embedding = nn.Embedding(input_dim, embed_dim)
#         self.positional_encoding = positional_encoding(max_seq_len, embed_dim)
#         self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
#         self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=6)
    
#     def forward(self, x):
#         x = self.embedding(x) + self.positional_encoding[:x.size(0), :]
#         x = self.transformer_encoder(x)
#         return x

# 使用示例
# input_dim = 1000  # 输入维度
# embed_dim = 512   # 嵌入维度
# max_seq_len = 100  # 最大序列长度
# model = TransformerModel(input_dim, embed_dim, max_seq_len)
# input_data = torch.randint(0, input_dim, (max_seq_len,))
# output = model(input_data.unsqueeze(1))


