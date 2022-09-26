# https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/pcgrad.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class PCGradOptim:
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        """
        clear the gradient of the parameters
        """

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        """
        update the parameters with the gradient
        """

        return self._optim.step()

    def pc_backward(self, objectives):
        """
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        """
        grads, shape = [], []
        for objective in objectives:
            grad, sh = [], []
            self.optimizer.zero_grad()
            objective.backward(retain_graph=True)

            for group in self._optim.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        grad.append(p.grad.clone()) 
                        sh.append(p.grad.shape)
            grad = torch.cat([g.flatten() for g in grad])
            grads.append(grad)
            shape.append(sh)
        
        pc_grads = copy.deepcopy(grads)
        for i, grad_i in enumerate(pc_grads):
            for j, grad_j in enumerate(grads):
                if i == j: continue
                dot_prod = (grad_i * grad_j).sum()
                if dot_prod < 0:
                    grad_i -= dot_prod * grad_j / (grad_j * grad_j).sum()

        pc_grads = sum(pc_grads)

        start, end = 0, 0
        grads= []
        for sh in shape[0]:
            end = start + torch.prod(torch.tensor(sh))
            grad = pc_grads[start: end].view(sh)
            grads.append(grad)

            start = end
        assert end == len(pc_grads)
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = grads[idx]
                    idx += 1
        assert idx == len(grads)