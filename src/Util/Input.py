from pymonntorch import Behavior, NeuronGroup
import torch


class InputBehavior(Behavior):

    def initialize(self, ng: NeuronGroup) -> None:
        self.spike = self.parameter('spikes', required=True)
        self.start_index = self.parameter('start_index')

        ng.spike = self.spike[20000:20000 + ng.size,
                              ng.network.iteration - 1]  # important
        ng.spike_counts = ng.spike

    def forward(self, ng: NeuronGroup) -> None:
        ng.spike = self.spike[20000:20000 + ng.size,
                              ng.network.iteration - 1]
        ng.spike_counts += ng.spike
