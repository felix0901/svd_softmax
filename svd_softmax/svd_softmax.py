import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import *
import time
from line_profiler import LineProfiler
import  cProfile
WINDOW_SIZE = 64
NUM_TOPK = 30
WORD_EMBEDDING_SIZE = 1280
HIDDEN_SIZE = 2560
NUM_WORDS = 200

class SvdSoftmax():
    def __init__(self,   window=WINDOW_SIZE, num_topk=NUM_TOPK):
        self.window = window
        self.num_topk = num_topk
    def set_weight(self, A, bias):
        U, S, V = torch.svd(A)
        self.B = torch.mm(U, torch.diag(S))
        self.V = V.t().contiguous()
        self.bias = bias.view(-1,1)
    def run_inference(self, x):
        x = torch.mm(self.V,x)
        z = torch.mm(self.B[:,:self.window], x[:self.window]) + self.bias

        #=======native=====
        # # z_idx = torch.argsort(z, descending=True, dim=0)[:self.num_topk]
        # _, z_idx = torch.topk(z, self.num_topk, dim=0, sorted=False)
        # z_idx = z_idx.view(-1)
        # # z_sel = torch.mm(self.B[z_idx,self.window:], x[self.window:])
        # # z[z_idx] += z_sel
        # z_sel = torch.mm(self.B[z_idx], x)
        # z[z_idx] = z_sel
        #================

        #====v1=========
        z = z / torch.sum(z, dim=0)
        z_sorted, z_idx = torch.sort(z, dim=0, descending=True)
        z_sorted = torch.cumsum(z_sorted, dim=0)
        multiplier = torch.from_numpy(np.array([(HIDDEN_SIZE-k)/(HIDDEN_SIZE-1) for k in range(HIDDEN_SIZE)])).view((HIDDEN_SIZE,-1))

        adjr2 = z_sorted * multiplier
        sel_num = torch.argmax(adjr2, dim=0)
        # plt.plot(adjr2)
        # plt.show()
        for i in range(len(sel_num)):
            pivot_point = sel_num[i]
            z_idx_used = z_idx[:pivot_point, i]
            x_used = x[:, i].view(-1,1)
            z_sel = torch.mm(self.B[z_idx_used], x_used).view(-1)
            z[z_idx_used, i] = z_sel
        # ========================

        ret = F.softmax(z, dim=0)
        return ret


class RegularSoftmax():
    def __init__(self):
        pass
    def set_weight(self, A, bias):
        self.A = A
        self.bias = bias.view(-1,1)
    def run_inference(self, x):
        x = torch.mm(self.A, x) + self.bias
        ret = F.softmax(x, dim=0)
        return ret


if __name__ == "__main__":
    input = torch.randn((WORD_EMBEDDING_SIZE, NUM_WORDS))
    weight = torch.randn((HIDDEN_SIZE, WORD_EMBEDDING_SIZE))
    bias = torch.randn((HIDDEN_SIZE,))
    act = SvdSoftmax()
    act_reg = RegularSoftmax()
    act.set_weight(weight,bias)
    act_reg.set_weight(weight, bias)
    #=====line profiler==============
    # lp = LineProfiler()
    # lp_wrapper = lp(act.run_inference)
    # lp_wrapper(input)
    # lp.print_stats()
    #=============================
    #======c profiler=============
    # cProfile.run("act_reg.run_inference(input)")
    #============================
    #======time time=============
    st = time.time()
    for _ in range(10):
        reg_softmax = act_reg.run_inference(input)
    reg_time = time.time() -st
    st = time.time()
    for _ in range(10):
        accl_softmax = act.run_inference(input)
    accl_time = time.time() -st

    norm_differnce = torch.mean(torch.abs(reg_softmax - accl_softmax))/torch.mean(accl_softmax)
    differnce = torch.mean(torch.abs(reg_softmax - accl_softmax))
    print(f'The mean difference is {differnce:.5f}, normalized mean difference is {norm_differnce:.5f}.')
    print(f'The original time is {reg_time:.5f}sec, our time is {accl_time:.5f}sec.')
    #============================