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

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class TrainAgent:
    def __init__(self, model, opt):
        self.model = model
        self.opt = opt
        
    def collect_trajs(self, num_traj=20, imitation=False):
        trajs = []
        for i in range(num_traj):
            game = Game()
            traj = []
            while not game.end_game:
                s = game.get_s().to(dev)[None, None, ...]
                p = self.model(s).squeeze()
                a = torch.multinomial(p, num_samples=1).item()
                if not imitation:
                    r = game.step(a)
                    traj.append([s, a, p, r])
                else:
                    a_greedy = game.get_greedy_a()
                    a = a if np.random.random() < 0.5 else a_greedy
                    r = game.step(a)
                    traj.append([s, a_greedy, p, r])
            trajs.append(traj)
        return trajs

    def compute_traj_loss(self, traj, imitation):
        loss = 0
        tot_r = 0
        for s, a, p, r in traj:
            tot_r += r
        reward_to_go = tot_r
        
        for s, a, p, r in traj:
            if not imitation:
                loss -= torch.log(p[a]) * reward_to_go
            else:
                target = torch.zeros(p.shape).to(dev)
                target[a] = 1
                loss += F.mse_loss(p, target)
            reward_to_go -= r
            
        return loss
    
    def compute_batch_loss(self, trajs, imitation):
        tot_loss = 0
        for traj in trajs:
            loss = self.compute_traj_loss(traj, imitation)
            tot_loss += loss
            
        tot_loss /= len(trajs)
        return tot_loss
    
    def batch_ascent(self, trajs, imitation=False):
        self.model.train()
        loss = self.compute_batch_loss(trajs, imitation)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss
    
    @torch.no_grad()
    def evalute(self, num_traj=20, imitation=False):
        trajs = self.collect_trajs(num_traj=num_traj)
        tot_r = 0
        for traj in trajs:
            traj_r = 0
            for s, a, p, r in traj:
                traj_r += r
            tot_r += traj_r
        tot_r /= num_traj
        print(tot_r)
    
    def train(self, epochs=100, checkpoint=10, imitation=False):
        losses = []
        for epoch in tqdm(range(epochs)):
            trajs = self.collect_trajs(imitation=imitation)
            loss = self.batch_ascent(trajs, imitation=imitation)
            loss = -loss.item()
            losses.append(loss)
            if epoch % checkpoint == 0:
                self.evalute(imitation=imitation)
        return losses