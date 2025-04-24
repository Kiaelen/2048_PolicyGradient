import numpy as np
import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from env import Game
from pg_model import PolicyNet
from tqdm import tqdm
from torch import optim
import random
import inspect

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

NUM_TRAJ = 40
NUM_UPDATES = 8
HIGH_SCORE = 10000

A, B = 0.5, 3e-3

def get_model(model, path=None, entire=False):
    if inspect.isclass(model):
        model = model()
    if path != None:
        if not entire:
            model.load_state_dict(torch.load(path, weights_only=True))
        else:
            model = torch.load(path, weights_only=False)
    model = model.to(dev)
    opt = optim.AdamW(model.parameters())
    return model, opt

class TrainAgent:
    def __init__(self, policy, value, opt, expert_policy=None):
        self.policy = policy
        self.value = value
        self.opt = opt
        self.max_score = 0
        self.expert_policy = expert_policy
        
    def collect_trajs(self, num_traj=NUM_TRAJ, offline=False, expert_policy=None):
        trajs = []
        for i in range(num_traj):
            if i != 0 and not offline:
                game = Game(random.choice(self.start_s))
            else:
                game = Game()
                
            traj = []
            
            while not game.end_game:
                if not offline:
                    if np.random.rand() < B:
                        self.start_s.append(Game(game))
                    s = game.get_s().to(dev)
                    p = self.policy(s[None, ...]).squeeze()
                    a = torch.multinomial(p, num_samples=1).item()
                    r = game.step(a)
                    traj.append([s, torch.tensor(a).to(dev), p.detach(), torch.tensor(r).to(dev)])
                else:
                    if expert_policy == None:
                        copy_game = Game(game)
                        if np.random.rand() < A:
                            traj.append(copy_game)
                        a = game.get_greedy_a()
                        game.step(a)
                    else:
                        copy_game = Game(game)
                        if np.random.rand() < A:
                            traj.append(copy_game)
                        s = game.get_s(old_ver=True).to(dev)
                        p = expert_policy(s[None, ...]).squeeze()
                        a = torch.multinomial(p, num_samples=1).item()
                        game.step(a)
            if not offline:
                trajs.append(traj)
            else:
                trajs += traj
            
            self.max_score = max(self.max_score, game.score)
            
        return trajs
    
    def compute_traj_loss(self, traj, type):
        loss = 0
        
        if type == "V": # Value loss
            tot_r = 0.
            for s, a, pt, r, p, v in traj:
                tot_r += r
            reward_to_go = tot_r
            
            for s, a, pt, r, p, v in traj:
                loss += F.mse_loss(v, torch.tensor([reward_to_go], dtype=v.dtype, device=dev))
                reward_to_go -= r
                
        if type == "P": # Policy loss
            A_gae = 0
            length = len(traj)
            
            for [s, a, pt, r, p, v], idx in zip(traj[::-1], range(length-1, -1, -1)):
                A_gae = A_gae * self.lamda + r - v.item()
                assert v.shape[0] == 1, "v shape error"
                if idx != length - 1:
                    A_gae += traj[idx+1][5].item()
                    assert traj[idx+1][5].shape[0] == 1, "v shape error"
                
                ratio = p[a] / pt[a]
                clip_ratio = ratio.clamp(1 - self.epsilon, 1 + self.epsilon)
                minn = torch.min(ratio * A_gae, clip_ratio * A_gae)
                
                loss -= minn
            
        return loss
    
    def compute_batch_loss(self, trajs):
        tot_loss = 0
        for traj in trajs:
            loss = 0
            loss += self.compute_traj_loss(traj, "V")
            loss += self.compute_traj_loss(traj, "P")
            
            tot_loss += loss
            
        tot_loss /= len(trajs)
        return tot_loss
    
    def augment(self, trajs):
        s_, a_, pt_, r_ = [], [], [], []
        for traj in trajs:
            for s, a, p, r in traj:
                s_.append(s)
                a_.append(a)
                pt_.append(p)
                r_.append(r)
                
        s_ = torch.stack(s_)
        a_ = torch.stack(a_)
        pt_ = torch.stack(pt_)
        r_ = torch.stack(r_)
        
        p_ = self.policy(s_)
        v_ = self.value(s_)
        
        aug_trajs = []
        idx = 0
        for traj in trajs:
            aug_traj = []
            for s, a, p, r in traj:
                aug_traj.append([s, a, p, r, p_[idx], v_[idx]])
                idx += 1
            aug_trajs.append(aug_traj)
        
        assert idx == s_.shape[0], "Shape dismatch"
        
        return aug_trajs
    
    def compute_rotation_loss(self, trajs):
        s_, s_rot, rot_dir = [], [], []
        for traj in trajs:
            for s, a, p, r in traj:
                s_.append(s)
                deg = np.random.randint(1, 4)
                rot_dir.append(deg)
                s_rot.append(torch.rot90(s, deg, (1, 2)))
                
        s_ = torch.stack(s_)
        s_rot = torch.stack(s_rot)
        
        p_ = self.policy(s_)
        pr_ = self.policy(s_rot)
        
        prs_ = []
        for i in range(len(pr_)):
            prs = torch.zeros(4)
            prs[:4-rot_dir[i]] = pr_[i, rot_dir[i]:]
            prs[rot_dir[i]:] = pr_[i, :4-rot_dir[i]]
            prs_.append(prs)
        
        prs_ = torch.stack(prs_)
        
        loss = F.mse_loss(p_, prs_)
        return loss
            
    def batch_ascent(self, trajs, num_updates=NUM_UPDATES, sym_w=0):
        self.policy.train()
        self.value.train()
        
        for _ in range(num_updates):
            aug_trajs = self.augment(trajs)
            loss = self.compute_batch_loss(aug_trajs)
            loss += sym_w * self.compute_rotation_loss(trajs)
                
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
        return loss
    
    @torch.no_grad()
    def evalute(self, num_traj=NUM_TRAJ):
        tot_score = 0
        for _ in range(num_traj):
            game = Game()
            while not game.end_game:
                s = game.get_s().to(dev)[None, ...]
                p = self.policy(s).squeeze()
                a = torch.multinomial(p, num_samples=1).item()
                r = game.step(a)
            tot_score += game.score
            
        tot_score /= num_traj
        return tot_score
    
    def collect_rand(self, num_s=5000):
        start_s = []
        for _ in range(num_s):
            start_s.append(Game(rand=True))
        return start_s
    
    def train(self, epochs=100, lamda=0.5, epsilon=0.2):
        self.lamda = lamda
        self.epsilon = epsilon
        
        losses = []
        checkpoint = epochs // 20
        
        # Collect offline data
        print("Collecting offline data")
        num_greedy_s = num_expert_s = num_rand_s = 0
        self.start_s = self.collect_trajs(num_traj=50, offline=True)
        num_greedy_s = len(self.start_s)
        if self.expert_policy != None:
            self.start_s += self.collect_trajs(num_traj=50, offline=True, expert_policy=self.expert_policy)
            num_expert_s = len(self.start_s) - num_greedy_s
        
        self.start_s += self.collect_rand()
        num_rand_s = len(self.start_s) - num_greedy_s - num_expert_s
        print(f"Complete with " \
                + f"{num_greedy_s} greedy trajectory starting states, " \
                + f"{num_expert_s} expert trajectory starting states " \
            + f"and {num_rand_s} random starting states\n" \
            + f"This is {len(self.start_s)} starting states in total")
        
        # PPO
        print("Start PPO")
        for epoch in tqdm(range(epochs)):
            trajs = self.collect_trajs()
            loss = self.batch_ascent(trajs, sym_w=0.2)
            
            loss = -loss.item()
            losses.append(loss)
            if epoch % checkpoint == 0:
                print(f"Average score: {self.evalute()}, # of starting states: {len(self.start_s)}")
        print("Finished!")
        return losses