#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import parl
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BC(parl.Algorithm):
    def __init__(self,
                 model,
                 actor_lr=None,
                 critic_lr=None):
        """ BC algorithm
            Args:
                model(parl.Model): forward network of actor and critic.
        """
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        self.model = model.to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.model.get_actor_params(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.model.get_critic_params(), lr=critic_lr)

    def predict(self, obs):
        act_mean, _ = self.model.policy(obs)
        action = torch.tanh(act_mean)
        return action

    def sample(self, obs):
        act_mean, act_log_std = self.model.policy(obs)
        normal = Normal(act_mean, act_log_std.exp())
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        return action, normal  

    def BClearn(self, obs, ref_obs,ref_agent):
        #actor learn
        action,action_distribution = self.sample(obs)
        with torch.no_grad():
            ref_action = ref_agent.alg.predict(ref_obs)
        actor_loss = - action_distribution.log_prob(ref_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            action_now,action_distribution = self.sample(obs)
            r_q1,r_q2 = ref_agent.alg.model.value(ref_obs,action_now)
        q1,q2 = self.model.value(obs,action_now)
        critic_loss = F.mse_loss(q1,r_q1)+F.mse_loss(q2,r_q2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item(),actor_loss.item()
