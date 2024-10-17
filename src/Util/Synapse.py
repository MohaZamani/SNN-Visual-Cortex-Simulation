from pymonntorch import Behavior, SynapseGroup
import torch
import random


class DDFSynapse(Behavior):
    # DDF : Dirac Delta Function

    def initialize(self, sg: SynapseGroup) -> None:
        self.connection_mode = self.parameter(
            "connection_mode", default="full")

        if self.connection_mode == 'full':
            self.__init_full(sg)
        elif self.connection_mode == 'fixed_probability':
            self.__init_fixed_probability(sg)
        elif self.connection_mode == 'fixed_count':
            self.__init_fixed_count(sg)
        else:
            raise ValueError("unexpected conncetion_mode parameter")

    def __init_full(self, sg: SynapseGroup) -> None:
        self.with_scaling = self.parameter('with_scaling', default=False)
        self.mean = self.parameter('mean', default=10)
        self.std = self.parameter('std', default=1)
        sg.W = sg.matrix(f'normal(mean={self.mean}, std={self.std})')

    def __init_fixed_probability(self, sg: SynapseGroup) -> None:
        self.with_scaling = self.parameter('with_scaling', default=True)
        self.connection_prob = self.parameter('connection_prob', default=0.5)
        self.mean = self.parameter(
            'mean', default=10) / sg.src.size * self.connection_prob if self.with_scaling else 1
        self.std = self.parameter('std', default=1) / \
            sg.src.size * self.connection_prob if self.with_scaling else 1

        is_connected = sg.matrix('uniform') <= self.connection_prob
        sg.W = sg.matrix(0)

        sg.W[is_connected] = sg.matrix(
            f'normal(mean={self.mean}, std={self.std})')[is_connected]

    def __init_fixed_count(self, sg: SynapseGroup) -> None:
        # optional, we can remove scaling in this connection pattern
        self.with_scaling = self.parameter('with_scaling', default=True)
        self.connection_count = self.parameter(
            'connection_count', default=int(sg.src.size / 5))
        self.mean = self.parameter(
            'mean', default=10) / self.connection_count if self.with_scaling else 1
        self.std = self.parameter('std', default=1) / \
            self.connection_count if self.with_scaling else 1

        sg.W = sg.matrix(0)
        for i in range(sg.dst.size):
            pre_neurons = random.sample(
                range(sg.src.size), self.connection_count)

            sg.W[pre_neurons, i] = torch.normal(
                self.mean, self.std, (len(pre_neurons), ), dtype=sg.W.dtype)

    def forward(self, sg: SynapseGroup) -> None:
        if 'excitatory' in sg.tags:
            sg.dst.I += torch.matmul(sg.src.spike.float(), sg.W.float())
        if 'inhibitory' in sg.tags:
            sg.dst.I -= 0.08 * torch.matmul(sg.src.spike.float(), sg.W.float())
