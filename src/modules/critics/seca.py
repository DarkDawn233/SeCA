import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SeCACritic(nn.Module):
    def __init__(self, scheme, args):
        super(SeCACritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.output_type = "q"

        # Set up network layers
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim * self.n_agents * self.n_actions
        self.hid_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
            self.hyper_w_v = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.embed_dim, self.embed_dim))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.hid_dim, self.hid_dim))
            self.hyper_w_v = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.hid_dim, self.hid_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.hid_dim)

        self.hyper_b_2 = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                               nn.ReLU(),
                               nn.Linear(self.hid_dim, 1))
        
        self.hyper_b_v = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                               nn.ReLU(),
                               nn.Linear(self.hid_dim, 1))

    def forward(self, act, states):
        """
        act:    [bs, ep_len, n_agents, n_actions]
        states: [bs, ep_len, state_dim]
        """
        bs = states.size(0)
        states = states.reshape(-1, self.state_dim)
        action_probs = act.reshape(-1, 1, self.n_agents * self.n_actions)

        w1 = self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents * self.n_actions, self.hid_dim)
        b1 = b1.view(-1, 1, self.hid_dim)

        h = torch.relu(torch.bmm(action_probs, w1) + b1)

        w_final = self.hyper_w_final(states)
        w_final = w_final.view(-1, self.hid_dim, 1)

        h2 = torch.bmm(h, w_final)

        b2 = self.hyper_b_2(states).view(-1, 1, 1)

        h2 = h2.view(bs, -1, 1)
        b2 = b2.view(bs, -1, 1)
        h2 = h2 + b2
        if self.args.critic_q:
            return h2
        else:
            w_v = self.hyper_w_v(states)
            w_v = w_v.view(-1, self.hid_dim, 1)
            h_v = torch.bmm(h, w_v)
            b_v = self.hyper_b_v(states)
            v = h_v.view(bs, -1, 1) + b_v.view(bs, -1, 1)
            return h2, v









