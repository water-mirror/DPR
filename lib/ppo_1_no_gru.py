import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.data import Data, DataLoader

class Net_Generic(nn.Module):
    
    def __init__(self, lr, input_dim, hidden, output_dim):
        super(Net_Generic, self).__init__()
        self.lr = lr
        self.fcs = []
        self.l1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, output_dim)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class GATConv(MessagePassing):
    
    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super(GATConv, self).__init__(aggr='add')
        self.negative_slope = negative_slope
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.attn = nn.Linear(2 * out_channels, out_channels)

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

# class Attention_Pointer(nn.Module):
#     def __init__(self, hidden_size):
#         super(Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.vt = nn.Linear(hidden_size, 1, bias=False)

#     def forward(self, x1, x2):
#         wx1 = self.W1(x1)
#         wx2 = self.W2(x2)
#         u_i = self.vt(torch.tanh(wx1 + wx2)).squeeze(-1)
#         return u_i

# class Attention(nn.Module):
#     def __init__(self, hidden_size):
#         super(Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.W = nn.Linear(hidden_size, hidden_size, bias=False)

#     def forward(self, x1, x2):
#         wx1 = self.W(x1)
#         att = torch.matmul(wx1, x2.transpose(1,2))
#         return att

class Net_Actor(nn.Module):

    def __init__(self, lr, embedding_dim, node_dim):
        super(Net_Actor, self).__init__()
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.node_dim = node_dim
        
        self.W_in = torch.nn.Linear(embedding_dim, node_dim, bias=False)
        self.W_z = torch.nn.Linear(node_dim * 2, node_dim, bias=True)
        self.W_r = torch.nn.Linear(node_dim * 2, node_dim, bias=True)
        self.W_h = torch.nn.Linear(node_dim * 2, node_dim, bias=True)
        
        self.gat_0 = GATConv(embedding_dim, node_dim)
        self.gat_n = GATConv(node_dim, node_dim)
        self.gat_1 = GATConv(node_dim, node_dim)
        self.gat_2 = GATConv(node_dim, node_dim)
        self.lin_prob = torch.nn.Linear(node_dim, 1)
        self.lin_sisr_mean = torch.nn.Linear(node_dim, 1)
        self.lin_sisr_std = torch.nn.Linear(node_dim, 1)
        self.sisr_activation = torch.nn.ELU(alpha=1.0, inplace=False)

    def forward(self, data, n_nodes):
        state_, input_ = data.state_, data.input_
        state_ = state_.reshape([-1, self.node_dim])
        input_ = input_.reshape([state_.size(0), -1, self.embedding_dim])
        input_ = self.W_in(input_)
        input_ = torch.mean(input_, dim=1)
        
        z = torch.cat([state_, input_], dim=-1)
        z = torch.sigmoid(self.W_z(z))
        r = torch.cat([state_, input_], dim=-1)
        r = torch.sigmoid(self.W_r(r))
        h = torch.cat([r * state_, input_], dim=-1)
        h = torch.tanh(self.W_h(h))
        h = (1-z) * state_ + z * h
        h = torch.unsqueeze(h,1)
        
        x, e_r0, e_r1, e_n = data.x, data.edge_index_r0, data.edge_index_r1, data.edge_index_n
        gat_0_out = self.gat_0(x, e_n)
        gat_1_out = self.gat_1(gat_0_out, e_r1)
        gat_2_out = self.gat_1(gat_1_out, e_r1)
        gat_3_out = self.gat_n(gat_2_out, e_n)+gat_0_out
        gat_4_out = self.gat_2(gat_3_out, e_r0)
        gat_5_out = self.gat_2(gat_4_out, e_r0)
        gat_x_out = self.gat_n(gat_5_out, e_n)+gat_3_out
        x = gat_x_out.reshape((-1, n_nodes, self.node_dim))
        
        scores = torch.squeeze(self.lin_prob(x[:,1:]), -1)
        probs = F.softmax(scores, dim=-1)
        distrib_p = Categorical(probs)
        
        sisr_params_a = torch.squeeze(self.lin_sisr_mean(x[:,1:]),-1)
        sisr_params_b = torch.squeeze(torch.abs(self.lin_sisr_std(x[:,1:])),-1)
        sisr_params_a = self.sisr_activation(sisr_params_a)+2
        sisr_params_b = self.sisr_activation(sisr_params_b)+2
        distrib_s = Beta(sisr_params_a, sisr_params_b)
        
        h = torch.squeeze(h,1)
        return h, distrib_p, distrib_s

class Memory:
    
    def __init__(self):
        self.action_node = []
        self.action_sisr = []
        self.states_x = []
        self.states_e_r0 = []
        self.states_e_r1 = []
        self.states_e_n = []
        self.states_state_ = []
        self.states_input_ = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.action_node[:]
        del self.action_sisr[:]
        del self.states_x[:]
        del self.states_e_r0[:]
        del self.states_e_r1[:]
        del self.states_e_n[:]
        del self.states_state_[:]
        del self.states_input_[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    
    def __init__(self, lr, embedding_dim, node_dim, critic_hidden, n_anchors):
        super(ActorCritic, self).__init__()
        self.actor = Net_Actor(lr, embedding_dim, node_dim)
        self.critic = Net_Generic(lr, node_dim, critic_hidden, 1)
        self.n_anchors = n_anchors
    
    def act(self, data, n_nodes):
        h, distrib_p, distrib_s = self.actor(data, n_nodes)
        action_node = distrib_p.sample((self.n_anchors,)).transpose(0,1)
        action_sisr = distrib_s.sample().detach()
        return h, action_node, action_sisr, distrib_p, distrib_s
    
    def evaluate(self, memory_state, memory_node, memory_sisr, n_nodes):
        h, distrib_p, distrib_s = self.actor(memory_state, n_nodes)
        indices = torch.arange(memory_node.size(0)).repeat(self.n_anchors)
        indices = indices.reshape([-1,memory_node.size(0)]).transpose(0,1)
        
        action_logprobs_1 = [distrib_p.log_prob(memory_node[:,i]).cpu().detach().numpy() \
                             for i in range(self.n_anchors)]
        action_logprobs_1 = torch.Tensor(action_logprobs_1).transpose(0,1)
        action_logprobs_2 = distrib_s.log_prob(memory_sisr)
        action_logprobs = action_logprobs_1.to(action_logprobs_2.device)+action_logprobs_2[indices, memory_node]
        action_logprobs = torch.sum(action_logprobs, 1)
        dist_entropy = distrib_p.entropy()[indices] + distrib_s.entropy()[indices, memory_node]
        state_value = self.critic(h)
        return action_logprobs, state_value.squeeze_(-1), dist_entropy

class Agent:
    
    def __init__(self, lr, betas, gamma, K_epochs, eps_clip, train_batch_size,
                 embedding_dim, node_dim, critic_hidden, n_anchors, device="cpu:0"):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.train_batch_size = train_batch_size
        self.n_anchors = n_anchors
        
        self.policy = ActorCritic(lr,
                                  embedding_dim,
                                  node_dim,
                                  critic_hidden,
                                  n_anchors).to(self.device)
        self.policy_old = ActorCritic(lr,
                                  embedding_dim,
                                  node_dim,
                                  critic_hidden,
                                  n_anchors).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()
    
    def load(self, save_path):
        self.policy.load_state_dict(torch.load(save_path))
        self.policy_old.load_state_dict(torch.load(save_path))
    
    def save(self, save_path):
        torch.save(self.policy.state_dict(), save_path)
    
    def update(self, memory, n_nodes):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = np.concatenate(rewards)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).float().to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        rewards.unsqueeze_(1)
        
        d_old_states_x = torch.stack(memory.states_x).to(self.device).detach()
        d_old_states_e_r0 = torch.stack(memory.states_e_r0).to(self.device).detach()
        d_old_states_e_r1 = torch.stack(memory.states_e_r1).to(self.device).detach()
        d_old_states_e_n = torch.stack(memory.states_e_n).to(self.device).detach()
        d_old_states_state_ = torch.stack(memory.states_state_).to(self.device).detach()
        d_old_states_input_ = torch.stack(memory.states_input_).to(self.device).detach()
        d_old_action_node = torch.stack(memory.action_node).unsqueeze_(1).to(self.device).detach()
        d_old_action_sisr = torch.stack(memory.action_sisr).to(self.device).detach()
        d_old_logprobs = torch.stack(memory.logprobs).unsqueeze_(1).to(self.device).detach()
        
        dataset = []
        for i in range(d_old_states_x.size(0)):
            data = Data(x=d_old_states_x[i],
                        edge_index_r0=d_old_states_e_r0[i],
                        edge_index_r1=d_old_states_e_r1[i],
                        edge_index_n=d_old_states_e_n[i],
                        state_=d_old_states_state_[i],
                        input_=d_old_states_input_[i],
                        actions_node=d_old_action_node[i],
                        actions_sisr=d_old_action_sisr[i],
                        logprobs=d_old_logprobs[i],
                        rewards=rewards[i])
            dataset.append(data)
        loader = DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True)
        
        for _ in range(self.K_epochs):
            for batch in loader:
                tmp_batch_sisr = batch.actions_sisr.reshape([batch.actions_node.size(0),-1])
                logprobs, state_v, dist_entropy = self.policy.evaluate(batch, batch.actions_node, tmp_batch_sisr, n_nodes)
                dist_entropy = torch.mean(dist_entropy,1)
                ratios = torch.exp(logprobs - batch.logprobs)

                advantages = batch.rewards - state_v.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                
                # 这三个loss可以用参数协调权重
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_v, batch.rewards) - 0*dist_entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())