import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable

from model.utils import log_normal_density

#obs_space = 10
#msg_action_space = 3
#ctr_action_space = 2

class MessageActor(nn.Module):
    def __init__(self, frames, msg_action_space):
        super(MessageActor, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(2*msg_action_space))

        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128*32, 256)
        self.act_fc2 =  nn.Linear(256+2+2, 128)
        self.actor1 = nn.Linear(128, msg_action_space)
        self.actor2 = nn.Linear(128, msg_action_space)

    def forward(self, x, goal, speed):
        """
            returns message_action, log_action_prob
        """
        # action
        a = F.relu(self.act_fea_cv1(x))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))

        a = torch.cat((a, goal, speed), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean1 = F.sigmoid(self.actor1(a))
        mean2 = F.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return action, logprob, mean


class ControlActor(nn.Module):
    def __init__(self, obs_space, ctr_action_space):
        super(ControlActor, self).__init__()
        self.act_fc1 = nn.Linear(obs_space, 64)
        self.act_fc2 = nn.Linear(64, 128)
        self.mu = nn.Linear(128, ctr_action_space)
        self.mu.weight.data.mul_(0.1)
        # torch.log(std)
        self.logstd = nn.Parameter(torch.zeros(ctr_action_space))

    def forward(self, x):
        """
            returns value estimation, action, log_action_prob
        """
        # action
        act = self.act_fc1(x)
        act = F.tanh(act)
        act = self.act_fc2(act)
        act = F.tanh(act)
        mean = self.mu(act)  # N, num_actions
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)

        return action, logprob, mean

class CNNPolicy(nn.Module):
    def __init__(self, obs_space, msg_action_space, ctr_action_space, frames, n_agent):
        super(CNNPolicy, self).__init__()

	self.n_agent = n_agent
        self.msg_actor = MessageActor(frames, msg_action_space)
	self.m2c_cv = nn.Conv1d(n_agent, obs_space, kernel_size=1)
	self.m2c_pj = nn.Linear(msg_action_space, 1)
        self.ctr_actor = ControlActor(obs_space, ctr_action_space)

        # value
        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128*32, 256)
        self.crt_fc2 = nn.Linear(256+2+2, 128)
        self.critic = nn.Linear(128, 1)


    def forward(self, x, goal, speed):
        """
            returns value estimation, action, log_action_prob
        """
	print(goal.shape)
	print(speed.shape)
        # message action
        msg, _, _ = self.msg_actor(x, goal, speed)
# x: obss (n_batch, n_agent, n_feature), msg: (n_batch, n_agent, n_msg_dim)
	print(msg.shape)

	msg_expanded = msg.unsqueeze(1).repeat(1, self.n_agent, 1, 1).view(msg.size(0), self.n_agent, -1)
	print(msg_expanded.shape)

        ctr_input_cv = self.m2c_cv(msg_expanded.permute(0,2,1)).permute(0, 2, 1)
# input: (n_batch, n_agent, n_msg_dim*(n_agent-1))
	print(ctr_input_cv.shape)  # Debug shape

	ctr_input = self.m2c_pj(ctr_input_cv).squeeze(-1)
	print(str_input.shape)

        action, logprob, mean = self.ctr_actor(ctr_input)
# action (n_batch, n_agent, action_dim)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # value
        v = F.relu(self.crt_fea_cv1(x))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))
        v = torch.cat((v, goal, speed), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return v, action, logprob, mean

    def evaluate_actions(self, x, goal, speed, action):
        v, _, _, mean = self.forward(x, goal, speed)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy
