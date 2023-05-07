from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
)
from gym import Env
import numpy as np
import torch
from utils import to_np, to_torch

class BaseEnv(Env):
    def __init__(self, args):
        super(BaseEnv, self).__init__()
        self.args = args
        self.pid = None
        self.sample_idx = 0
        # TODO obs space and action space
        self.reward_list = []
        self.stl_reward_list = []
        self.acc_reward_list = []
        self.history = []
    
    @abstractmethod
    def next_state(self, x, u):
        pass

    # @abstractmethod
    def dynamics(self, x0, u, include_first=False):
        args = self.args
        t = u.shape[1]
        x = x0.clone()
        segs = []
        if include_first:
            segs.append(x)
        for ti in range(t):
            new_x = self.next_state(x, u[:, ti])
            segs.append(new_x)
            x = new_x
        return torch.stack(segs, dim=1)

    @abstractmethod
    def init_x_cycle(self):
        pass
    
    @abstractmethod
    def init_x(self):
        pass
    
    @abstractmethod
    def generate_stl(self):
        pass 
    
    @abstractmethod
    def generate_heur_loss(self):
        pass
    
    @abstractmethod
    def visualize(self):
        pass

    #@abstractmethod
    def step(self):
        pass
    
    #@abstractmethod
    # def reset(self):
        # pass
    def reset(self):
        N = self.args.num_samples
        if self.sample_idx % N == 0:
            self.x0 = self.init_x(N)
            self.indices = torch.randperm(N)
        self.state = to_np(self.x0[self.indices[self.sample_idx % N]])
        self.sample_idx += 1
        self.t = 0
        if len(self.history)>self.args.nt:
            segs_np = np.stack(self.history, axis=0)
            segs = to_torch(segs_np[None, :])
            self.reward_list.append(np.sum(self.generate_reward_batch(segs_np)))
            self.stl_reward_list.append(self.stl_reward(segs)[0, 0])
            self.acc_reward_list.append(self.acc_reward(segs)[0, 0])
        self.history = [np.array(self.state)]
        return self.state
    
    def get_rewards(self):
        if len(self.reward_list)==0:
            return 0, 0, 0
        else:
            return self.reward_list[-1], self.stl_reward_list[-1], self.acc_reward_list[-1]

    @abstractmethod
    def generate_reward_batch(self, state):
        pass

    #@abstractmethod
    def generate_reward(self, state):
        if self.args.stl_reward or self.args.acc_reward:
            last_one = (self.t+1) >= self.args.nt
            if last_one:
                segs = to_torch(np.stack(self.history, axis=0)[None, :])
                if self.args.stl_reward:
                    return self.stl_reward(segs)[0, 0]
                elif self.args.acc_reward:
                    return self.acc_reward(segs)[0, 0]
                else:
                    raise NotImplementError
            else:
                return np.zeros_like(0)
        else:
            return self.generate_reward_batch(state[None, :])[0]

    def stl_reward(self, segs):
        score = self.stl(segs, self.args.smoothing_factor)[:, :1]
        reward = to_np(score)
        return reward
    
    def acc_reward(self, segs):
        score = (self.stl(segs, self.args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        reward = 100 * to_np(score)
        return reward

    def print_stl(self):
        print(self.stl)
        self.stl.update_format("word")
        print(self.stl)

    def my_render(self):
        if self.pid==0:
            self.render(None)
    
    def test(self):
        for trial_i in range(self.num_trials):
            obs = self.test_reset()
            trajs = [self.test_state()]
            for ti in range(self.nt):
                u = solve(obs)
                obs, reward, done, di = self.test_step(u)
                trajs.append(self.test_state())
        
        # save metrics result