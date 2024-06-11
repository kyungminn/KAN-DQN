import argparse
import random
import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import collections
import random
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from convkan import *

from omegaconf import OmegaConf
from hydra import compose, initialize
from dotmap import DotMap
import wandb
import warnings
import ale_py

import csv
import datetime
import torch._dynamo
torch._dynamo.config.suppress_errors = True
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

    
    
class PreprocessEnv:
    def __init__(self, env, imgsize, device):
        self.env = env
        self.env = GrayScaleObservation(self.env)
        self.env = ResizeObservation(self.env, shape=(imgsize, imgsize))
        self.env = FrameStack(self.env, num_stack=4)
        self.metadata = env.metadata
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4, 1, imgsize, imgsize), dtype=np.float32)
        self.action_space = env.action_space
        self.device=device

    def _preproces_obs(self, obs):
        obs = torch.tensor(np.array(obs, dtype=np.float32)).unsqueeze(1)
        obs = obs / 255
        return obs.to(self.device)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._preproces_obs(obs), info

    def step(self, action):
        obs, score, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        reward = np.sign(score)
        info['score'] = torch.tensor(score).to(self.device)
        return self._preproces_obs(obs), torch.tensor(reward).to(self.device), torch.tensor(done).to(self.device), info
    
    def close(self):
        self.env.close()


class VecEnv:
    def __init__(self, env_list, num_eval_envs, imgsize, device):
        self.env_list = env_list
        self.num_eval_envs = num_eval_envs
        self.imgsize = imgsize
        self.device = device
        self.seeds = self._generate_seeds()
    
    def _generate_seeds(self):
        seeds = {env_name: [random.randint(0, 10000) for _ in range(self.num_eval_envs)] for env_name in self.env_list}
        return seeds
    
    def reset(self):
        # Init env
        envs = {}
        observations = {}
        infos = {}
        for env_name in self.env_list:
            envs[env_name] = []
            obs_list = []
            info_list = []
            for i in range(self.num_eval_envs):
                env = gym.make(f'ALE/{env_name}-v5', full_action_space=True)
                env = PreprocessEnv(env, self.imgsize, self.device)
                obs, info = env.reset(seed=self.seeds[env_name][i])
                envs[env_name].append(env)
                obs_list.append(obs)
                info_list.append(info)
            observations[env_name] = torch.stack(obs_list)
            infos[env_name] = info_list
        self.envs = envs
        return observations, infos


    def step(self, actions):
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        for env_name, env_list in self.envs.items():
            obs_list = []
            reward_list = []
            done_list = []
            info_list = []
            if env_name not in actions.keys():
                continue
            for env, action in zip(env_list, actions[env_name]):
                obs, reward, done, info = env.step(action)
                obs_list.append(obs)
                reward_list.append(reward)
                done_list.append(done)
                info_list.append(info)
            if len(obs_list)==0:
                continue
            else:
                observations[env_name] = torch.stack(obs_list)
                rewards[env_name] = reward_list
                dones[env_name] = done_list
                infos[env_name] = info_list
        return observations, rewards, dones, infos
    
    
    def eval_metric(self, q_net, eval_epsilon):
        observations, _ = self.reset()
        total_scores = {env_name: [[] for _ in range(len(env_list))] for env_name, env_list in self.envs.items()}
        scores_done = {env_name: [] for env_name in self.env_list}
        
        while any(len(env_list) > 0 for env_list in self.envs.values()):
            actions = {}
            for env_name, obs in observations.items():
                actions[env_name] = q_net.sample_action(obs, eval_epsilon).tolist()
            observations, rewards, dones, infos = self.step(actions)
            
            for env_name in observations.keys():
                done_indices = []
                for i, (info, reward, done) in enumerate(zip(infos[env_name], rewards[env_name], dones[env_name])):
                    total_scores[env_name][i].append(info['score'].item())
                    if done:
                        scores_done[env_name].append(np.sum(total_scores[env_name][i]))
                        done_indices.append(i)
                        
                # Remove done environments and their rewards
                for idx in reversed(done_indices):
                    del total_scores[env_name][idx]
                    del self.envs[env_name][idx]
        
        # Calculate average scores
        avg_scores = {env_name: np.mean(scores_done[env_name]) if scores_done[env_name] else 0.0 for env_name in self.env_list}
        return avg_scores
    
class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)
        
class Augmentation(nn.Module):
    def __init__(self, aug=False):
        super().__init__()
        self.layers = []
        if aug:
            self.layers.append(RandomShiftsAug(pad=4))
            self.layers.append(Intensity(scale=0.05))
            self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
            
    

class ReplayBuffer():
    def __init__(self, buffer_limit, obs_shape, max_n_step, gamma, device):
        self.buffer_limit = buffer_limit
        # self.buffer = collections.deque(maxlen=buffer_limit)
        self.max_n_step = max_n_step+1
        self.n_step_buffer = collections.deque(maxlen=self.max_n_step)
        self.gamma = gamma

        self.observations = np.zeros((self.buffer_limit, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((self.buffer_limit, 1), dtype=np.int64)
        self.done_masks =  np.zeros((self.buffer_limit, 1), dtype=np.float32)
        self.n_step_rew_sums = np.zeros((self.buffer_limit, 1), dtype=np.float32)
        self.n_step_offsets = np.zeros((self.buffer_limit, 1), dtype=np.float32)

        self.device = device
        self.cursor = 0
        self.buffer_size = 0

    def put(self, transition):
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.max_n_step:
            return
            
        obs, action, _, _ = self.n_step_buffer[0]
        self.observations[self.cursor] = np.array(obs)*255.0
        self.actions[self.cursor] = action
        
        _, _, G, done_mask = self.n_step_buffer[-1]
        n_step_offset = 0
        for i in reversed(range(self.max_n_step-1)):
            _, _, r, d = self.n_step_buffer[i]
            G = r + self.gamma*G*d
            done_mask = np.logical_and(done_mask, d)
            n_step_offset = 1 + n_step_offset*d
        
        self.done_masks[self.cursor] = done_mask
        self.n_step_rew_sums[self.cursor] = G
        self.n_step_offsets[self.cursor] = n_step_offset
        self.buffer_size = min(self.buffer_size+1, self.buffer_limit)
        self.cursor = (self.cursor + 1) % self.buffer_limit

    def sample(self, batch_size):
        assert self.buffer_size > 0
        if batch_size > self.buffer_size:
            print("WARNING | sample() called with insufficient number of samples")

        # Effective size is buffer_size - max_n_step.
        # Technically we have to wait until effective size = batch size.
        batch_idx = np.random.randint(self.buffer_size-self.max_n_step, size=batch_size)
        # If done is within max_n_steps, next_obs_idx doesn't matter anyway.
        next_obs_idx = (batch_idx + self.max_n_step) % self.buffer_size

        return torch.tensor(self.observations[batch_idx]/255.0, dtype=torch.float32, device=self.device), \
               torch.tensor(self.actions[batch_idx], device=self.device), \
               torch.tensor(self.n_step_rew_sums[batch_idx], device=self.device), \
               torch.tensor(self.observations[next_obs_idx]/255.0, dtype=torch.float32, device=self.device), \
               torch.tensor(self.done_masks[batch_idx], device=self.device), \
               torch.tensor(self.n_step_offsets[batch_idx], device=self.device) # Don't think we need this one, but keeping it just in case.
    
    @property
    def size(self):
        return self.buffer_size
    
    def clear(self):
        self.buffer_size = 0
        self.cursor = 0
        self.n_step_buffer.clear()
        

class DQN(nn.Module):
    def __init__(self, config):
        super(DQN, self).__init__()
        
        self.config=config
        self.device=self.config.device
        if self.config.model == 'orig_DQN':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=self.config.frame_size, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                nn.ReLU()
            )

            self.fc = nn.Sequential(
                nn.Linear(in_features=64*7*7 , out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=self.config.action_size)
            )
        elif self.config.model == 'KAN_DQN':
            self.conv = nn.Sequential(
                ConvKAN(in_channels=self.config.frame_size, out_channels=self.config.h_dim[0], kernel_size=8, stride=4),
                LayerNorm2D(self.config.h_dim[0]),
                ConvKAN(in_channels=self.config.h_dim[0], out_channels=self.config.h_dim[1], kernel_size=4, stride=2),
                LayerNorm2D(self.config.h_dim[1]),
                ConvKAN(in_channels=self.config.h_dim[1], out_channels=self.config.h_dim[2], kernel_size=3, stride=1),
                LayerNorm2D(self.config.h_dim[2])
            ).to(self.config.device)
            self.fc = nn.Sequential(
                KANLinear(in_features=self.config.h_dim[2]*7*7 , out_features=128),
                nn.LayerNorm(128),
                KANLinear(in_features=128, out_features=self.config.action_size)
            ).to(self.config.device)
            
        
    def forward(self, x):
        x = x.squeeze(2)
        if self.config.model == 'orig_DQN':
            conv_out = self.conv(x).view(x.size()[0],-1)
            return self.fc(conv_out)
        elif self.config.model == 'KAN_DQN':
            batch = x.shape[0]
            conv_out = self.conv(x).reshape(batch,-1)
            return self.fc(conv_out.unsqueeze(0)).squeeze(0)
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        batch_size = out.shape[0]
        random_actions = torch.randint(0, 18, (batch_size,)).to(self.device)
        greedy_actions = out.argmax(dim=1)
        epsilon_mask = (torch.rand(batch_size) < epsilon).long().to(self.device)
        actions = epsilon_mask * random_actions + (1 - epsilon_mask) * greedy_actions
        return actions
    
    def soft_update(self, net_target, tau):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            
            
def train(q, q_target, memory, aug_func, optimizer, batch_size, gamma, n_steps):
    s,a,r,s_prime,done_mask,n_step_offset = memory.sample(batch_size)
    s, s_prime = aug_func(s.squeeze(2)), aug_func(s_prime.squeeze(2))
    q_out = q(s)
    q_a = q_out.gather(1,a)
    max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
    target = r + (gamma**n_steps)*max_q_prime*done_mask
    loss = F.smooth_l1_loss(q_a, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def save_dict(config, checkpoint_dict, name):
    os.makedirs(f'./models/{config.project_name}/{config.group_name}/{config.exp_name}/seed{config.seed}', exist_ok=True)
    path = f'./models/{config.project_name}/{config.group_name}/{config.exp_name}/seed{config.seed}/{name}.pth'
    torch.save(checkpoint_dict, path)
    print("Checkpoint saved successfully at", path)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(config):
    wandb.init(project=config.project_name,
               entity=config.entity,
               config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
               group=config.group_name,
               name=config.wandb_run_name)
    
    q = torch.compile(DQN(config.dqn)).to(config.device)
    q_target = torch.compile(DQN(config.dqn)).to(config.device)
    q_target.load_state_dict(q.state_dict())
    print(f'number of Q_net params: {count_parameters(q)}')
    buffer = ReplayBuffer(config.buffer_limit, config.obs_shape, config.max_n_step, config.dqn.gamma, config.device)
    optimizer = optim.Adam(q.parameters(), lr=config.dqn.lr)
    aug_func = Augmentation(aug=config.aug)
    
    # init eval env
    eval_envs = VecEnv(env_list=config.env_list,
                       num_eval_envs=config.num_eval_envs,
                       imgsize=config.imgsize,
                       device=config.device)
    score_dict = {}
    for env_list in config.env_list:
        score_dict[f'{env_list}'] = []
    total_step=0
    for env_name in config.env_list:
        env = gym.make(f'ALE/{env_name}-v5', full_action_space=True)
        env = PreprocessEnv(env, config.imgsize, config.device)
        buffer.clear()
        step=0
        print(f'Start {env_name}')
        print(f'Memory size: {buffer.size}')
        while True:
            epsilon = config.init_epsilon - min(1, step/config.decay_steps)*(config.init_epsilon-config.min_epsilon)
            s, _ = env.reset(seed=config.seed)
            done = False
            r_list = []
            while not done:
                a = q.sample_action(s.to(config.device).unsqueeze(0), epsilon)      
                s_prime, r, done, info = env.step(a)
                done_mask = 0.0 if done else 1.0
                buffer.put((s.cpu(),a.cpu(),r.cpu(),done_mask))
                s = s_prime
                r_list.append(r)
            wandb.log({'sum_of_rewards': sum(r_list)}, step=total_step)
                
            print(f'Memory size: {buffer.size}')
            episode_length = len(r_list)
            if buffer.size >= config.min_buffer_size:
                for _ in tqdm(range(episode_length * config.dqn.replay_ratio)):
                    if step >= config.num_episodes_per_env:
                        break
                    q_loss = train(q, q_target, buffer, aug_func, optimizer, config.dqn.batch_size, config.dqn.gamma, config.max_n_step)
                    q.soft_update(q_target, config.dqn.tau)
                    step += 1
                    total_step += 1

                    if step%config.log_every==0 and step!=0:
                        log_dict = {}
                        log_dict['n_epi']=step
                        log_dict['q_loss']=q_loss
                        wandb.log(log_dict, step=total_step)
                        
                    if step%config.eval_every==0 and step!=0:
                        print('Start Eval')
                        eval_dict = {}
                        with torch.no_grad():
                            scores = eval_envs.eval_metric(q_net=q, eval_epsilon=config.eval_epsilon)
                            print('done!')
                            for e_name in config.env_list:
                                eval_dict[f'Eval {e_name} score'] = scores[f'{e_name}']
                                score_dict[f'{e_name}'].append(scores[f'{e_name}'])
                        wandb.log(eval_dict, step=total_step)

            if step >= config.num_episodes_per_env:
                break
        
        name = f'{env_name}'+'_'+str(step)
        checkpoint_dict={'q': q.state_dict(),
                        'q_target': q_target.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'score': score_dict}
        save_dict(config, checkpoint_dict, name)
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_path', type=str,    default='./configs')
    parser.add_argument('--config_name', type=str,    default='kan_dqn') 
    parser.add_argument('--overrides',   action='append', default=[])
    args = parser.parse_args()
    
    args = DotMap(vars(args))
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides
    
    # Hydra Compose
    initialize(version_base=None, config_path=config_path) 
    config = compose(config_name=config_name, overrides=overrides)
    print(OmegaConf.to_yaml(config))
    main(config)