import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

class Attention_Pointer(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x1, x2):
        wx1 = self.W1(x1)
        wx2 = self.W2(x2)
        u_i = self.vt(torch.tanh(wx1 + wx2)).squeeze(-1)
        return u_i

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x1, x2):
        wx1 = self.W(x1)
        att = torch.matmul(wx1, x2.transpose(1,2))
        return att

class Net_Actor(nn.Module):

    def __init__(self, lr, embedding_dim, node_dim, n_nodes):
        super(Net_Actor, self).__init__()
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.node_dim = node_dim
        self.n_nodes = n_nodes
        
        self.gat_0 = GATConv(embedding_dim, node_dim)
        self.gat_n = GATConv(node_dim, node_dim)
        self.gat_1 = GATConv(node_dim, node_dim)
        self.gat_2 = GATConv(node_dim, node_dim)
        self.lin_prob = torch.nn.Linear(node_dim, 1)
        
        self.attn_prev = Attention(node_dim)
        self.attn_next = Attention(node_dim)

    def forward(self, data):
        x, e_r0, e_r1, e_n = data.x, data.edge_index_r0, data.edge_index_r1, data.edge_index_n
        gat_0_out = self.gat_0(x, e_n)
        gat_n_out = self.gat_n(gat_0_out, e_n)
        gat_1_out = self.gat_1(gat_n_out, e_r1)
        gat_2_out = self.gat_1(gat_1_out, e_r1)
        gat_3_out = self.gat_1(gat_2_out, e_r1)
        gat_4_out = self.gat_1(gat_3_out, e_r1)
        gat_5_out = self.gat_n(gat_4_out, e_n)
        gat_6_out = self.gat_2(gat_5_out, e_r0)
        gat_7_out = self.gat_2(gat_6_out, e_r0)
        gat_8_out = self.gat_2(gat_7_out, e_r0)
        gat_9_out = self.gat_2(gat_8_out, e_r0)
        gat_x_out = self.gat_n(gat_9_out, e_n)
        x = gat_x_out.reshape((-1, self.n_nodes, self.node_dim))
        h = torch.mean(x, dim=1)
        
        score_prev = self.attn_prev(x[:,1:], x[:,1:])
        score_next = self.attn_next(x[:,1:], x[:,1:])
        score_prev[:,torch.arange(x.size(1)-1),torch.arange(x.size(1)-1)] = float("-inf")
        score_next[:,torch.arange(x.size(1)-1),torch.arange(x.size(1)-1)] = float("-inf")
        scores = torch.cat((score_prev, score_next), 1)
        scores = scores.reshape([-1, scores.size(1) * scores.size(2)])
        probs = F.softmax(scores, dim=-1)
        distrib = torch.distributions.categorical.Categorical(probs)
        
        return h, distrib

class Memory:
    
    def __init__(self):
        self.actions = []
        self.states_x = []
        self.states_e_r0 = []
        self.states_e_r1 = []
        self.states_e_n = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states_x[:]
        del self.states_e_r0[:]
        del self.states_e_r1[:]
        del self.states_e_n[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    
    def __init__(self, lr, embedding_dim, node_dim, n_nodes, critic_hidden):
        super(ActorCritic, self).__init__()
        self.actor = Net_Actor(lr, embedding_dim, node_dim, n_nodes)
        self.critic = Net_Generic(lr, node_dim, critic_hidden, 1)
    
    def act(self, data, memory):
        _, distribs = self.actor(data)
        action = distribs.sample()
        return action, distribs
    
    def evaluate(self, memory_state, memory_action):
        h, distrib = self.actor(memory_state)
        action_logprobs = distrib.log_prob(memory_action)
        dist_entropy = distrib.entropy()
        state_value = self.critic(h)
        return action_logprobs, state_value.squeeze(-1), dist_entropy

class Agent:
    
    def __init__(self, lr, betas, gamma, K_epochs, eps_clip, train_batch_size,
                 embedding_dim, node_dim, n_nodes, critic_hidden, device="cpu:0"):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.train_batch_size = train_batch_size
        
        self.policy = ActorCritic(lr,
                                  embedding_dim,
                                  node_dim,
                                  n_nodes,
                                  critic_hidden).to(self.device)
        self.policy_old = ActorCritic(lr,
                                  embedding_dim,
                                  node_dim,
                                  n_nodes,
                                  critic_hidden).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()
    
    def load(self, save_path):
        self.policy.load_state_dict(torch.load(save_path))
        self.policy_old.load_state_dict(torch.load(save_path))
    
    def save(self, save_path):
        torch.save(self.policy.state_dict(), save_path)
    
    def update(self, memory):
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
        rewards = rewards.unsqueeze(1)
        
        d_old_states_x = torch.stack(memory.states_x).to(self.device).detach()
        d_old_states_e_r0 = torch.stack(memory.states_e_r0).to(self.device).detach()
        d_old_states_e_r1 = torch.stack(memory.states_e_r1).to(self.device).detach()
        d_old_states_e_n = torch.stack(memory.states_e_n).to(self.device).detach()
        d_old_actions = torch.stack(memory.actions).unsqueeze(1).to(self.device).detach()
        d_old_logprobs = torch.stack(memory.logprobs).unsqueeze(1).to(self.device).detach()
        
        dataset = []
        for i in range(d_old_states_x.size(0)):
            data = Data(x=d_old_states_x[i],
                        edge_index_r0=d_old_states_e_r0[i],
                        edge_index_r1=d_old_states_e_r1[i],
                        edge_index_n=d_old_states_e_n[i],
                        actions=d_old_actions[i],
                        logprobs=d_old_logprobs[i],
                        rewards=rewards[i])
            dataset.append(data)
        loader = DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True)
        
        for _ in range(self.K_epochs):
            for batch in loader:
                logprobs, state_v, dist_entropy = self.policy.evaluate(batch, batch.actions)

                ratios = torch.exp(logprobs - batch.logprobs)

                advantages = batch.rewards - state_v.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                
                # 这三个loss可以用参数协调权重
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_v, batch.rewards) - 0.01*dist_entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())