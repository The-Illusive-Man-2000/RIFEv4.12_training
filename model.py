import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from warplayer import warp
from torchstat import stat
from torch.nn.parallel import DistributedDataParallel as DDP
from Flownet import *
import torch.nn.functional as F
from loss import *
from vgg import *
from ssim import SSIM

device = torch.device("cuda")
    
class Model:
    def __init__(self, local_rank=-1):
        self.flownet = FlownetCas()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-2)
        self.epe = EPE()
        self.lap = LapLoss()
        self.ss = SSIM()
        self.vgg = VGGPerceptualLoss().to(device)
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))), False)
        
    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))

    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, distill=False, timestep=0.5):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        scale = [8, 4, 2, 1]
        p = np.random.uniform(0, 1)
        if training:
            if p < 0.3:
                scale = [4, 2, 1, 1]
            elif p < 0.6:
                scale = [2, 1, 1, 1]
        flow, mask, merged, teacher_res, loss_cons = self.flownet(torch.cat((imgs, gt), 1), timestep, scale=scale, training=training, distill=distill)
        loss_l1 = 0
        for i in range(4):
            loss_l1 += (merged[i] - gt).abs().mean()
        loss_tea = (teacher_res[0][0] - gt).abs().mean() + ((teacher_res[1][0] ** 2 + 1e-6).sum(1) ** 0.5).mean() * 1e-5
        loss_cons += ((flow[-1] ** 2 + 1e-6).sum(1) ** 0.5).mean() * 1e-5
        loss_vgg = self.vgg(merged[3], gt).mean() - self.ss(merged[3], gt) * 0.1
        if training:
            self.optimG.zero_grad()
            loss_G = loss_vgg + loss_tea * 0.1 + loss_cons + loss_l1 * 0.1
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(self.flownet.parameters(), 1.0)
            self.optimG.step()
            flow_teacher = teacher_res[1][0]
        else:
            flow_teacher = flow[3]
        return merged[3], {
            'merged_tea': teacher_res[0][0],
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[3][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_cons': loss_cons,
            }
