
from __future__ import division
import torch
import numpy as np
import torch.nn as nn
from gym import spaces
import torch.nn.functional as F
from torch.autograd import Variable

from utils import norm_col_init, weights_init
from perception import NoisyLinear, BiRNN, AttentionLayer,FirstAwareBranch,SecAwareBranch,MLP,PredictionLayer


def build_model(obs_space, action_space, args, device):
    name = args.model

    if 'single-att' in name:
        model = A3C_Single(obs_space, action_space, args, device)
        if args.load_executor_dir:
            saved_state = torch.load(
                args.load_executor_dir,  
                map_location=lambda storage, loc: storage)
            model.load_state_dict(saved_state['model'], strict=False)  
    elif 'v1' in name:
        model = A3C_Single_FM(obs_space, action_space, args, device)
    elif 'single-fm' in name:
        model = A3C_Single_FM_v2(obs_space, action_space, args, device)
        if args.load_executor_dir:
            saved_state = torch.load(
                args.load_executor_dir,  
                map_location=lambda storage, loc: storage)
            model.load_state_dict(saved_state['model'], strict=False)        
    elif 'multi' in name:
        model = A3C_Multi(obs_space, action_space, args, device)
    elif 'fm-att' in name:
        model = A3C_Multi_FM(obs_space, action_space, args, device)
    model.train()
    return model


def wrap_action(self, action):
    action = np.squeeze(action)
    out = action * (self.action_high - self.action_low) / 2 + (self.action_high + self.action_low) / 2.0
    return out


def sample_action(mu_multi, sigma_multi, device, test=False):

    logit = mu_multi
    prob = F.softmax(logit, dim=-1)
    log_prob = F.log_softmax(logit, dim=-1)
    entropy = -(log_prob * prob).sum(-1, keepdim=True)
    if test:
        action = prob.max(-1)[1].data
        action_env = action.cpu().numpy()  # np.squeeze(action.cpu().numpy(), axis=0)
    else:
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))  # [num_agent, 1] # comment for sl slave
        action_env = action.squeeze(0)

    return action_env, entropy, log_prob


class ValueNet(nn.Module):
    def __init__(self, input_dim, head_name, num=1):
        super(ValueNet, self).__init__()
        if 'ns' in head_name:
            self.noise = True
            self.critic_linear = NoisyLinear(input_dim, num, sigma_init=0.017)
        else:
            self.noise = False
            self.critic_linear = nn.Linear(input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)

    def forward(self, x):
        value = self.critic_linear(x)
        return value

    def sample_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()


class AMCValueNet(nn.Module):
    def __init__(self, input_dim, head_name, num=1, device=torch.device('cpu')):
        super(AMCValueNet, self).__init__()
        self.head_name = head_name
        self.device = device

        if 'ns' in head_name:
            self.noise = True
            self.critic_linear = NoisyLinear(input_dim, num, sigma_init=0.017)
        if 'onlyJ' in head_name:
            self.noise = False
            self.critic_linear = nn.Linear(input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)
        else:
            self.noise = False
            self.critic_linear = nn.Linear(2 * input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)

            self.attention = AttentionLayer(input_dim, input_dim, device)
        self.feature_dim = input_dim

    def forward(self, x, goal):
        _, feature_dim = x.shape
        value = []

        coalition = x.view(-1, feature_dim)
        n = coalition.shape[0]

        feature = torch.zeros([self.feature_dim]).to(self.device)
        value.append(self.critic_linear(torch.cat([feature, coalition[0]])))
        for j in range(1, n):
            _, feature = self.attention(coalition[:j].unsqueeze(0))
            value.append(self.critic_linear(torch.cat([feature.squeeze(), coalition[j]])))  # delta f = f[:j]-f[:j-1]

        # mean and sum
        value = torch.cat(value).sum()

        return value.unsqueeze(0)

    def sample_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()
class AMCValueNetFM(nn.Module):
    def __init__(self, input_dim, head_name, num=1, device=torch.device('cpu')):
        super(AMCValueNetFM, self).__init__()
        self.head_name = head_name
        self.device = device

        if 'ns' in head_name:
            self.noise = True
            self.critic_linear = NoisyLinear(input_dim, num, sigma_init=0.017)
        if 'onlyJ' in head_name:
            self.noise = False
            self.critic_linear = nn.Linear(input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)
        else:
            self.noise = False
            self.critic_linear = nn.Linear(20, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)

            self.attention = AttentionLayer(input_dim, input_dim, device)
        self.feature_dim = input_dim

    def forward(self, x, goal):
        
        x=x.squeeze()
        return (torch.max(x)).unsqueeze(0)
        

    def sample_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()

class PolicyNet(nn.Module):
    def __init__(self, input_dim, action_space, head_name, device):
        super(PolicyNet, self).__init__()
        self.head_name = head_name
        self.device = device
        num_outputs = action_space.n

        if 'ns' in head_name:
            self.noise = True
            self.actor_linear = NoisyLinear(input_dim, num_outputs, sigma_init=0.017)
        else:
            self.noise = False
            self.actor_linear = nn.Linear(input_dim, num_outputs)

            # init layers
            self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.1)
            self.actor_linear.bias.data.fill_(0)

    def forward(self, x, test=False):
        mu = F.relu(self.actor_linear(x))
        sigma = torch.ones_like(mu)
        action, entropy, log_prob = sample_action(mu, sigma, self.device, test)
        return action, entropy, log_prob

    def sample_noise(self):
        if self.noise:
            self.actor_linear.sample_noise()
            self.actor_linear2.sample_noise()

class PolicyNetFM(nn.Module):
    def __init__(self, input_dim, action_space, head_name, device):
        super(PolicyNetFM, self).__init__()
        self.head_name = head_name
        self.device = device
        num_outputs = action_space.n
        self.out = PredictionLayer('binary', )
        if 'ns' in head_name:
            self.noise = True
        #     self.actor_linear = NoisyLinear(input_dim, num_outputs, sigma_init=0.017)
        else:
            self.noise = False
        #     self.actor_linear = nn.Linear(input_dim, num_outputs)

        #     # init layers
        #     self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.1)
        #     self.actor_linear.bias.data.fill_(0)

    def forward(self, x, test=False):
        # mu = F.relu(self.actor_linear(x))
        # sigma = torch.ones_like(mu)
        # action, entropy, log_prob = sample_action(mu, sigma, self.device, test)
        # return action, entropy, log_prob
        prob=self.out(x)
        complement = 1 - prob
        prob = torch.cat((complement, prob), dim=1)
        log_prob = torch.log(prob)
        entropy = -(log_prob * prob).sum(-1, keepdim=True)
        if test:
          action = prob.max(-1)[1].data
          action_env = action.cpu().numpy()  # np.squeeze(action.cpu().numpy(), axis=0)
        else:
          action = prob.multinomial(1).data
          log_prob = log_prob.gather(1, Variable(action))  # [num_agent, 1] # comment for sl slave
          action_env = action.squeeze(0)

        return action_env, entropy, log_prob
    def sample_noise(self):
        if self.noise:
          pass
            # self.actor_linear.sample_noise()
            # self.actor_linear2.sample_noise()

    def remove_noise(self):
        if self.noise:
          pass
            # self.actor_linear.sample_noise()
            # self.actor_linear2.sample_noise()

class EncodeBiRNN(torch.nn.Module):
    def __init__(self, dim_in, lstm_out=128, head_name='birnn_lstm', device=None):
        super(EncodeBiRNN, self).__init__()
        self.head_name = head_name

        self.encoder = BiRNN(dim_in, int(lstm_out / 2), 1, device, 'gru')

        self.feature_dim = self.encoder.feature_dim
        self.global_feature_dim = self.encoder.feature_dim
        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        x = inputs
        cn, hn = self.encoder(x)

        feature = cn  # shape: [bs, num_camera, lstm_dim]

        global_feature = hn.permute(1, 0, 2).reshape(-1)

        return feature, global_feature

class EncodeLinear(torch.nn.Module):
    def __init__(self, dim_in, dim_out=32, head_name='lstm', device=None):
        super(EncodeLinear, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out),
            nn.ReLU(inplace=True)
        )

        self.head_name = head_name
        self.feature_dim = dim_out
        self.train()

    def forward(self, inputs):
        x = inputs
        feature = self.features(x)
        return feature

class A3C_Single(torch.nn.Module):  # single vision Tracking
    def __init__(self, obs_space, action_spaces, args, device=torch.device('cpu')):
        super(A3C_Single, self).__init__()
        self.n = len(obs_space)
        obs_dim = obs_space[0].shape[1]
        
        lstm_out = args.lstm_out
        head_name = args.model

        self.head_name = head_name

        self.encoder = AttentionLayer(obs_dim, lstm_out, device)
        self.critic = ValueNet(lstm_out, head_name, 1)
        self.actor = PolicyNet(lstm_out, action_spaces[0], head_name, device)

        self.train()
        self.device = device

    def forward(self, inputs, test=False):
        data = Variable(inputs, requires_grad=True)
        _, feature = self.encoder(data)

        actions, entropies, log_probs = self.actor(feature, test)
        values = self.critic(feature)
        return values, actions, entropies, log_probs

    def sample_noise(self):
        self.actor.sample_noise()
        self.actor.sample_noise()

    def remove_noise(self):
        self.actor.remove_noise()
        self.actor.remove_noise()

class FM(nn.Module):
    '''Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    '''

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term
  
        


class A3C_Multi(torch.nn.Module):
    def __init__(self, obs_space, action_spaces, args, device=torch.device('cpu')):
        super(A3C_Multi, self).__init__()
        self.num_agents, self.num_targets, self.pose_dim = obs_space.shape

        lstm_out = args.lstm_out
        head_name = args.model
        self.head_name = head_name

        self.encoder = EncodeLinear(self.pose_dim, lstm_out, head_name, device)
        feature_dim = self.encoder.feature_dim

        self.attention = AttentionLayer(feature_dim, lstm_out, device)
        feature_dim = self.attention.feature_dim

        # create actor & critic
        self.actor = PolicyNet(feature_dim, spaces.Discrete(2), head_name, device)

        if 'shap' in head_name:
            self.ShapleyVcritic = AMCValueNet(feature_dim, head_name, 1, device)
        else:
            self.critic = ValueNet(feature_dim, head_name, 1)

        self.train()
        self.device = device

    def forward(self, inputs, test=False):
        pos_obs = inputs

        feature_target = Variable(pos_obs, requires_grad=True)
        feature_target = self.encoder(feature_target)  # num_agent, num_target, feature_dim

        feature_target = feature_target.reshape(-1, self.encoder.feature_dim).unsqueeze(0)  # [1, agent*target, feature_dim]
        feature, global_feature = self.attention(feature_target)  # num_agents, feature_dim
        feature = feature.squeeze()

        actions, entropies, log_probs = self.actor(feature, test)
        actions = actions.reshape(self.num_agents, self.num_targets, -1)

        if 'shap' not in self.head_name:
            values = self.critic(global_feature)
        else:
            values = self.ShapleyVcritic(feature, actions)  # shape [1,1]

        return values, actions, entropies, log_probs

    def sample_noise(self):
        self.actor.sample_noise()
        self.actor.sample_noise()

    def remove_noise(self):
        self.actor.remove_noise()
        self.actor.remove_noise()
        
class FirstOrderAware(nn.Module):
    def __init__(self,device = torch.device('cpu') ):
        super(FirstOrderAware, self).__init__()
        self.first_integrate = FirstAwareBranch(5, 10, device = device)

        self.first_aware = MLP(50,
                                (128,128,), activation='relu', l2_reg=0,
                                dropout_rate=0.1,
                                use_ln=True, init_std=0.0001, device=device)

        self.first_reweight = nn.Linear(
            128, 5, bias=False).to(device)


        self.ln = nn.LayerNorm(50).to(device)
        self.regularization_weight = []
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.first_aware.named_parameters()),
            l2=0)

        self.add_regularization_weight(self.first_reweight.weight, l2=0)
        
    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))
        
    def forward(self,x):
        
        first_out = self.first_integrate(x)
        first_out = self.ln(first_out)
        first_out = self.first_aware(first_out)
        m_first= self.first_reweight(first_out)
        
        return m_first

class SecondOrderAware(nn.Module):
    def __init__(self,device = torch.device('cpu') ):
        super(SecondOrderAware, self).__init__()
        self.sec_integrate = SecAwareBranch()
        
        self.sec_aware = MLP(50,
                                (128,), activation='relu', l2_reg=0,
                                dropout_rate=0.1,
                                use_ln=True, init_std=0.0001, device=device)    
        
        self.sec_reweight = nn.Linear(
            128, 5, bias=False).to(device)        


        self.ln = nn.LayerNorm(50).to(device)
        self.regularization_weight = []
        
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.sec_aware.named_parameters()),
            l2=0)

        self.add_regularization_weight(self.sec_reweight.weight, l2=0)
        
    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))
        
    def forward(self,x):
        
        sec_out = self.sec_integrate(x)
        sec_out = self.ln(sec_out)
        sec_out = self.sec_aware(sec_out)
        m_sec = self.sec_reweight(sec_out)

        return m_sec
    
class Gate(nn.Module):
    def __init__(self,input_dim, num_expert):
        super(Gate, self).__init__()
        self.ln1 = nn.Linear(input_dim,128)
#         self.ln1.weight.data = norm_col_init(self.ln1.weight.data, 0.1)
#         self.ln1.bias.data.fill_(0)
        self.norm1 = nn.LayerNorm(128)
        self.relu = nn.ReLU()
        self.ln2 = nn.Linear(128,num_expert)
#         self.ln2.weight.data = norm_col_init(self.ln2.weight.data, 0.1)
#         self.ln2.bias.data.fill_(0)
        self.norm2 = nn.LayerNorm(num_expert)
     
    def forward(self, inputs):
        return self.ln2(self.relu(self.ln1(inputs)))
    
class A3C_Single_FM_v2(torch.nn.Module):  # single vision Tracking
    def __init__(self, obs_space, action_spaces, args, device=torch.device('cpu')):
        super(A3C_Single_FM_v2, self).__init__()
        self.n = len(obs_space)
        obs_dim = obs_space[0].shape[1]
        
        lstm_out = args.lstm_out
        head_name = args.model
        self.num_agents = obs_space.shape[0]
        self.num_targets = obs_space.shape[1]

        
        self.head_name = head_name


        self.critic = ValueNet(lstm_out, head_name, 1)
        self.actor = PolicyNet(lstm_out, action_spaces[0], head_name, device)
        
#         self.fm = FM()
        self.linear = nn.Linear(1, 10)

        # init layers
        self.linear.weight.data = norm_col_init(self.linear.weight.data, 0.1)
        self.linear.bias.data.fill_(0)        
#         self.first_integrate = FirstAwareBranch(4, 10, device = device)

#         self.first_aware = MLP(40,
#                                 (128,), activation='relu', l2_reg=0,
#                                 dropout_rate=0.1,
#                                 use_ln=True, init_std=0.0001, device=device) 
        self.ln = nn.LayerNorm(40).to(device)
        self.ln2 = nn.LayerNorm(128).to(device)
        
        self.sec_integrate = SecAwareBranch()
        
        self.sec_aware = MLP(40,
                                (128,128,), activation='relu', l2_reg=0,
                                dropout_rate=0.1,
                                use_ln=True, init_std=0.0001, device=device)   
        
        self.train()
        
    def forward(self, x, test=False):
        inputs=torch.reshape(x,(x.shape[0]*x.shape[1],4,1))
#         covered = ((torch.abs(inputs[:, 2]) < (45.0/180.0)) & ((inputs[:, 2] + inputs[:,3])!= 0)).int()
#         covered = covered/2.0
#         inputs = torch.cat((inputs, covered.unsqueeze(1)), dim=1)
#         inputs = inputs.unsqueeze(-1)
        inputs = Variable(inputs, requires_grad=True)
        
        embed = self.linear(inputs)
        first_out = self.sec_integrate(embed)
        first_out = self.ln(first_out)
        first_out = self.sec_aware(first_out)
        first_out = first_out.reshape(x.shape[0],x.shape[1],-1)
        feature = first_out.sum(dim=1)
        feature = self.ln2(feature)
        
        actions, entropies, log_probs = self.actor(feature, test)
        values = self.critic(feature)        

        return values, actions, entropies, log_probs      

    def sample_noise(self):
        self.actor.sample_noise()
        self.actor.sample_noise()

    def remove_noise(self):
        self.actor.remove_noise()
        self.actor.remove_noise()    
        
class A3C_Multi_FM(torch.nn.Module):
    def __init__(self, obs_space, action_spaces, args, device=torch.device('cpu')):
        super(A3C_Multi_FM, self).__init__()
        self.num_agents, self.num_targets, self.pose_dim = obs_space.shape

        lstm_out = args.lstm_out
        head_name = args.model
        self.head_name = head_name

        self.encoder = EncodeLinear(self.pose_dim, lstm_out, head_name, device)
        feature_dim = self.encoder.feature_dim

        self.attention = AttentionLayer(feature_dim, lstm_out, device)
        feature_dim = self.attention.feature_dim

        # create actor & critic
        self.actor = PolicyNetFM(feature_dim, spaces.Discrete(2), head_name, device)
        self.embedding = nn.Embedding(self.num_agents+self.num_targets+58 , 10,device=device)
        self.embedding.weight.data.uniform_(-.1, .1)
        torch.nn.init.xavier_normal_(self.embedding.weight.data, gain=1e-3)

        self.embedding2 = nn.Embedding(self.num_agents+self.num_targets+58 , 1,device=device)
        self.embedding2.weight.data.uniform_(-.1, .1)
        torch.nn.init.xavier_normal_(self.embedding2.weight.data, gain=1e-3)
        
        self.num_experts = 4
        self.k = 1
        self.gate = Gate(50,self.num_experts)
        self.experts = nn.ModuleList([FirstOrderAware(device) for _ in range(self.num_experts)])
        self.fm = FM()
        if 'shap' in head_name:
            self.ShapleyVcritic = AMCValueNetFM(feature_dim, head_name, 1, device)
        else:
            self.critic = ValueNet(feature_dim, head_name, 1)

        self.train()
        self.device = device

    def forward(self, inputs, test=False):
#         inputs=torch.reshape(inputs,(self.num_agents*self.num_targets,4))
#         inputs = Variable(inputs, requires_grad=True)

#         pos_obs=[]
#         target_offset = torch.tensor(self.num_agents)
#         angle_offset = torch.tensor(self.num_agents+self.num_targets-1)
#         distance_offset = torch.tensor(self.num_agents+self.num_targets+35)
#         visible_offset =torch.tensor(self.num_agents+self.num_targets+56)
#         for obs in inputs:
#           sensor = obs[0] * self.num_agents
#           target = obs[1] * self.num_targets
#           angle = torch.ceil(torch.abs(obs[2]) *(180.0/5.0))
#           distance = torch.clip(torch.ceil(obs[3] *(2000.0/100)),max=20)
#           if (torch.abs(obs[2]) < (45.0/180.0)) and (obs[3] <= (200.0/2000.0)):
#             visible=torch.tensor(1)
#           else:
#             visible=torch.tensor(0)

#           pos_obs.append(torch.Tensor([sensor,target + target_offset,angle + angle_offset,distance + distance_offset,visible+visible_offset]))
        
#         pos_obs = torch.stack(pos_obs).type(torch.int32).to(self.device)
        # Reshape inputs to batch processing dimensions
        inputs = torch.reshape(inputs, (self.num_agents * self.num_targets, 4))
        inputs = Variable(inputs, requires_grad=True)

        # Define constants
        target_offset = torch.tensor(self.num_agents, device=self.device)
        angle_offset = torch.tensor(self.num_agents + self.num_targets - 1, device=self.device)
        distance_offset = torch.tensor(self.num_agents + self.num_targets + 35, device=self.device)
        visible_offset = torch.tensor(self.num_agents + self.num_targets + 56, device=self.device)

        # Batch processing calculations
        sensors = inputs[:, 0] * self.num_agents
        targets = inputs[:, 1] * self.num_targets + target_offset
        angles = torch.ceil(torch.abs(inputs[:, 2]) * (180.0 / 5.0)) + angle_offset
        distances = torch.clip(torch.ceil(inputs[:, 3] * (2000.0 / 100)), max=20) + distance_offset

        # Compute visibility for all inputs in one operation
        visibility_conditions = (torch.abs(inputs[:, 2]) < (45.0 / 180.0)) & (inputs[:, 3] <= (200.0 / 2000.0))
        visibilities = torch.where(visibility_conditions, torch.tensor(1, device=self.device), torch.tensor(0, device=self.device)) + visible_offset

        # Combine all computed values into a single tensor
        pos_obs = torch.stack([sensors, targets, angles, distances, visibilities], dim=1).type(torch.int32).to(self.device)

        embed = self.embedding(pos_obs).to(self.device)
#         print("Embed: ",embed.shape)
#         m_final = self.first(embed)
#         print("m_final: ",m_final.shape)
        gate_values = self.gate(embed.reshape(self.num_agents*self.num_targets,-1))  # Compute gating scores
        topk_values, topk_indices = torch.topk(gate_values, self.k, dim=1)  # Select top-k experts

        normalized_topk_values = F.softmax(topk_values, dim=1)       
        m_final = torch.zeros(embed.size(0), 5, device = embed.device)
        
        for i in range(self.k):
            expert_index = topk_indices[:, i]
            expert_output = torch.stack([self.experts[idx](embed[b].unsqueeze(0)) for b, idx in enumerate(expert_index)])
            m_final += normalized_topk_values[:, i].unsqueeze(1) * expert_output.squeeze(1)

        refined = embed * m_final.unsqueeze(-1)

        feature = self.fm(refined)

        embed2 = self.embedding2(pos_obs).to(self.device)
        embed2 = embed2.reshape(self.num_agents*self.num_targets,1,-1)
        embed2 = embed2*m_final.unsqueeze(1)
        logit = torch.sum(embed2, dim=-1, keepdim=False)

        feature+=logit
        
        #feature = self.ln2(self.relu(self.linear(feature)))
        
        actions, entropies, log_probs = self.actor(feature, test)
        actions = actions.reshape(self.num_agents, self.num_targets, -1)

        # if 'shap' not in self.head_name:
        #     values = self.critic(global_feature)
        # else:
        values = self.ShapleyVcritic(feature, actions)  # shape [1,1]

        return values, actions, entropies, log_probs

    def sample_noise(self):
        self.actor.sample_noise()
        self.actor.sample_noise()

    def remove_noise(self):
        self.actor.remove_noise()
        self.actor.remove_noise()

