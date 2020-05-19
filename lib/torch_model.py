import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.data import Data, DataLoader

class GATConv(MessagePassing):
    
    def __init__(self, in_channels, out_channels, negative_slope=0.2, device="cpu:0"):
        super(GATConv, self).__init__(aggr='add')
        self.negative_slope = negative_slope
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.attn = nn.Linear(2 * out_channels, out_channels)
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index_i, size_i):
        # x_j -> x_i
        x = torch.cat([x_i, x_j], dim=-1)
        alpha = self.attn(x)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)
        return x_j * alpha

    def update(self, aggr_out):
        return aggr_out

class Net_Actor(nn.Module):

    def __init__(self, lr, embedding_dim, node_dim, n_nodes, device="cpu:0"):
        super(Net_Actor, self).__init__()
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.node_dim = node_dim
        self.n_nodes = n_nodes
        
        self.W_in = torch.nn.Linear(embedding_dim, node_dim, bias=False)
        self.W_z = torch.nn.Linear(node_dim * 2, node_dim, bias=False)
        self.W_r = torch.nn.Linear(node_dim * 2, node_dim, bias=False)
        self.W_h = torch.nn.Linear(node_dim * 2, node_dim, bias=False)
        
        self.gat_0 = GATConv(embedding_dim, node_dim, device=device)
        self.gat_1 = GATConv(node_dim, node_dim, device=device)
        # self.gat_2 = GATConv(node_dim, node_dim)
        self.lin_prob = torch.nn.Linear(node_dim, 1)
        self.lin_sisr = torch.nn.Linear(node_dim, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, data, state_, input_):
        # state_ = (-1, node_dim)
        # input_ = (-1, N_remove, 16)
        input_ = self.W_in(input_)
        input_ = torch.mean(input_, dim=1)
        
        z = torch.cat([state_, input_], dim=-1)
        z = torch.sigmoid(self.W_z(z))
        r = torch.cat([state_, input_], dim=-1)
        r = torch.sigmoid(self.W_r(r))
        h = torch.cat([r * state_, input_], dim=-1)
        h = torch.tanh(self.W_h(h))
        h = (1-z) * state_ + z * h
        h.unsqueeze_(1)
        
        x, edge_index_0, edge_index_1 = data.x, data.edge_index_0, data.edge_index_1
        gat_0_out = self.gat_0(x, edge_index_0)
        # gat_1_out = self.gat_1(gat_0_out, edge_index_0)
        # gat_2_out = self.gat_2(gat_1_out, edge_index_1)
        # x = gat_2_out.reshape((-1, self.n_nodes, self.node_dim))
        gat_1_out = self.gat_1(gat_0_out, edge_index_1)
        x = gat_1_out.reshape((-1, self.n_nodes, self.node_dim))
        x = x * h
        
        prob = self.lin_prob(x)
        sisr = self.lin_sisr(x)
        prob = prob.reshape((-1, self.n_nodes))
        sisr = sisr.reshape((-1, self.n_nodes))
        prob = F.softmax(prob[:,1:], dim=-1)
        sisr = torch.sigmoid(sisr)
        
        h = h.reshape((-1, self.node_dim))
        
        return prob, sisr, h
    
class Net_Generic(nn.Module):
    
    def __init__(self, lr, input_dim, hidden1, hidden2, output_dim, device="cpu:0"):
        super(Net_Generic, self).__init__()
        self.lr = lr
        self.fcs = []
        self.l1 = nn.Linear(input_dim, hidden1)
        self.l2 = nn.Linear(hidden1, hidden2)
        self.l3 = nn.Linear(hidden2, output_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device(device)
        self.to(self.device)
    
    def forward(self, state):
        # state is a torch.Tensor
        x = state.to(self.device)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class Agent(object):
    
    def __init__(self,
                 lr_actor,
                 lr_critic,
                 embedding_dim,
                 node_dim,
                 n_nodes,
                 critic_hidden,
                 gamma=0.99,
                 device="cpu:0"):
        self.gamma  = gamma
        self.actor = Net_Actor(lr_actor, embedding_dim, node_dim, n_nodes, device=device)
        self.critic = Net_Generic(lr_critic, node_dim, critic_hidden, critic_hidden, 1, device=device)
    
    def choose_action(self, observation, n):
        
        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
            return torch.index_select(a, dim, order_index)
        
        data, h, input_ = observation
        prob, sisr, h_ = self.actor(data, h, input_)
        anchors = torch.distributions.Categorical(prob).sample((n,)).transpose(0,1)
        coefficients = sisr[tile(torch.arange(prob.size(0)), 0, n), torch.reshape(anchors,[-1])].reshape([-1,n])
        log_probs = prob[tile(torch.arange(prob.size(0)), 0, n), torch.reshape(anchors,[-1])].reshape([-1,n])
        log_probs = torch.sum(log_probs,1)
        result = (anchors.detach().cpu().numpy()+1,
                  coefficients.detach().cpu().numpy(),
                  h_,
                  torch.log(log_probs))
        return result
    
    def learn(self, h, reward, h_, log_prob):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        
        critic_value = torch.reshape(self.critic(h), (-1,))
        critic_value_ = torch.reshape(self.critic(h_), (-1,))
        
        td = self.gamma * critic_value_ + reward -critic_value
        actor_loss = td * log_prob
        critic_loss = td**2
        loss = torch.mean(actor_loss)+torch.mean(critic_loss)
        
        loss.backward(retain_graph=True)
        self.actor.optimizer.step()
        self.critic.optimizer.step()
    
    def save(self, actor_path, critic_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    def load(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))