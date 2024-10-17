from pymonntorch import NeuronGroup, SynapseGroup
from matplotlib import pyplot as plt
import numpy as np
import torch


def spike_times_plot(ng: NeuronGroup, s=10):
    plt.rcParams.update({'font.size': 6})
    plt.figure(figsize=(4.5, 3.5))
    plt.scatter(ng['spike'][0][ng['spike', 0][:, 1] == 1] * 0.01,
                ng['spike'][0][ng['spike', 0][:, 1] == 1], s=s)
    plt.xlabel('time', fontsize=12)
    plt.ylabel('iteration Nnumber', fontsize=12)
    plt.title('spike times', fontsize=12)
    plt.show()


def input_and_dynamic_plot(ng: NeuronGroup, iter_num: int = 100):
    # plot 1:
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ng['v', 0][:iter_num, :])
    plt.hlines(y=-70, xmin=-10, xmax=iter_num+5,
               linestyles='--', color='g', label='resting potential')
    plt.hlines(y=-40, xmin=-10, xmax=iter_num+5,
               linestyles='--', color='r', label='threshold')
    plt.xlabel('iteration')
    plt.ylabel('Voltage')
    plt.legend()

    # plot 2:
    plt.subplot(1, 2, 2)
    plt.plot(ng['I', 0][:iter_num, :])
    plt.xlabel('iteration')
    plt.ylabel('input current')

    plt.show()


def plot_weights_changes_in_stdp(sg: SynapseGroup):
    x = np.arange(0, 700)

    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(figsize=(5.5, 3.3))

    ax.plot(x, sg['W'][0][:, 0][:, 0], lw=1, label='W11')
    ax.plot(x, sg['W'][0][:, 0][:, 1], lw=1, label='W12')
    ax.plot(x, sg['W'][0][:, 1][:, 0], lw=1, label='W21')
    ax.plot(x, sg['W'][0][:, 1][:, 1], lw=1, label='W22')
    ax.plot(x, sg['W'][0][:, 2][:, 0], lw=1, label='W31')
    ax.plot(x, sg['W'][0][:, 2][:, 1], lw=1, label='W32')
    ax.plot(x, sg['W'][0][:, 3][:, 0], lw=1, label='W41')
    ax.plot(x, sg['W'][0][:, 3][:, 1], lw=1, label='W42')

    plt.legend()
    plt.show()


def plot_weights_changes_of_specific_output__in_stdp(sg1: SynapseGroup, sg2: SynapseGroup, out_neuron_num: int):
    x = np.arange(0, 690)

    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(figsize=(5.5, 3.3))

    ax.plot(x, sg1['W'][0][:690, 0][:, out_neuron_num], lw=1, label='W11')
    # ax.plot(x, sg['W'][0][:, 0][:, 1], lw=1, label='W12')
    ax.plot(x, sg1['W'][0][:690, 1][:, out_neuron_num], lw=1, label='W21')
    # ax.plot(x, sg['W'][0][:, 1][:, 1], lw=1, label='W22')
    ax.plot(x, sg1['W'][0][:690, 2][:, out_neuron_num], lw=1, label='W31')
    # ax.plot(x, sg['W'][0][:, 2][:, 1], lw=1, label='W32')
    ax.plot(x, sg1['W'][0][:690, 3][:, out_neuron_num], lw=1, label='W41')
    # ax.plot(x, sg['W'][0][:, 3][:, 1], lw=1, label='W42')

    ax.plot(x, sg2['W'][0][:690, 0][:, out_neuron_num], lw=1, label='W51')
    # ax.plot(x, sg['W'][0][:, 0][:, 1], lw=1, label='W12')
    ax.plot(x, sg2['W'][0][:690, 1][:, out_neuron_num], lw=1, label='W61')
    # ax.plot(x, sg['W'][0][:, 1][:, 1], lw=1, label='W22')
    ax.plot(x, sg2['W'][0][:690, 2][:, out_neuron_num], lw=1, label='W71')
    # ax.plot(x, sg['W'][0][:, 2][:, 1], lw=1, label='W32')
    ax.plot(x, sg2['W'][0][:690, 3][:, out_neuron_num], lw=1, label='W81')
    # ax.plot(x, sg['W'][0][:, 3][:, 1], lw=1, label='W42')

    plt.legend()
    plt.show()


def new_plot_weights_changes_of_specific_output__in_stdp(sg1: SynapseGroup, sg2: SynapseGroup, sg3: SynapseGroup, sg4: SynapseGroup, sg5: SynapseGroup, out_neuron_num: int):
    x = np.arange(0, 690)

    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(figsize=(5.5, 3.3))

    ax.plot(x, sg1['W'][0][:690, 0][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 0][:, 1], lw=1, label='W12')
    ax.plot(x, sg1['W'][0][:690, 1][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 1][:, 1], lw=1, label='W22')
    ax.plot(x, sg1['W'][0][:690, 2][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 2][:, 1], lw=1, label='W32')
    ax.plot(x, sg1['W'][0][:690, 3][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 3][:, 1], lw=1, label='W42')

    ax.plot(x, sg2['W'][0][:690, 0][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 0][:, 1], lw=1, label='W12')
    ax.plot(x, sg2['W'][0][:690, 1][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 1][:, 1], lw=1, label='W22')
    ax.plot(x, sg2['W'][0][:690, 2][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 2][:, 1], lw=1, label='W32')
    ax.plot(x, sg2['W'][0][:690, 3][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 3][:, 1], lw=1, label='W42')

    ax.plot(x, sg3['W'][0][:690, 0][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 0][:, 1], lw=1, label='W12')
    ax.plot(x, sg3['W'][0][:690, 1][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 1][:, 1], lw=1, label='W22')
    ax.plot(x, sg3['W'][0][:690, 2][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 2][:, 1], lw=1, label='W32')
    ax.plot(x, sg3['W'][0][:690, 3][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 3][:, 1], lw=1, label='W42')

    ax.plot(x, sg4['W'][0][:690, 0][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 0][:, 1], lw=1, label='W12')
    ax.plot(x, sg4['W'][0][:690, 1][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 1][:, 1], lw=1, label='W22')
    ax.plot(x, sg4['W'][0][:690, 2][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 2][:, 1], lw=1, label='W32')
    ax.plot(x, sg4['W'][0][:690, 3][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 3][:, 1], lw=1, label='W42')

    ax.plot(x, sg5['W'][0][:690, 0][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 0][:, 1], lw=1, label='W12')
    ax.plot(x, sg5['W'][0][:690, 1][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 1][:, 1], lw=1, label='W22')
    ax.plot(x, sg5['W'][0][:690, 2][:, out_neuron_num], lw=1)
    # ax.plot(x, g['W'][0][:, 2][:, 1], lw=1, label='W32')
    ax.plot(x, sg5['W'][0][:690, 3][:, out_neuron_num], lw=1)
    # ax.plot(x, sg['W'][0][:, 3][:, 1], lw=1, label='W42')

    plt.legend()
    plt.show()


def plot_weights_changes_in_stdp_three_output(sg: SynapseGroup):
    x = np.arange(0, 700)

    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(figsize=(5.5, 3.3))

    ax.plot(x, sg['W'][0][:, 0][:, 0], lw=1, label='W11')
    ax.plot(x, sg['W'][0][:, 0][:, 1], lw=1, label='W12')
    ax.plot(x, sg['W'][0][:, 0][:, 2], lw=1, label='W13')
    ax.plot(x, sg['W'][0][:, 1][:, 0], lw=1, label='W21')
    ax.plot(x, sg['W'][0][:, 1][:, 1], lw=1, label='W22')
    ax.plot(x, sg['W'][0][:, 1][:, 2], lw=1, label='W23')
    ax.plot(x, sg['W'][0][:, 2][:, 0], lw=1, label='W31')
    ax.plot(x, sg['W'][0][:, 2][:, 1], lw=1, label='W32')
    ax.plot(x, sg['W'][0][:, 2][:, 2], lw=1, label='W33')
    ax.plot(x, sg['W'][0][:, 3][:, 0], lw=1, label='W41')
    ax.plot(x, sg['W'][0][:, 3][:, 1], lw=1, label='W42')
    ax.plot(x, sg['W'][0][:, 3][:, 2], lw=1, label='W43')

    plt.legend()
    plt.show()


def plot_weights_changes_in_stdp_three_output(sg: SynapseGroup):
    x = np.arange(0, 700)

    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(figsize=(5.5, 3.3))

    ax.plot(x, sg['W'][0][:, 0][:, 0], lw=1, label='W11')
    ax.plot(x, sg['W'][0][:, 0][:, 1], lw=1, label='W12')
    ax.plot(x, sg['W'][0][:, 0][:, 2], lw=1, label='W13')
    ax.plot(x, sg['W'][0][:, 1][:, 0], lw=1, label='W21')
    ax.plot(x, sg['W'][0][:, 1][:, 1], lw=1, label='W22')
    ax.plot(x, sg['W'][0][:, 1][:, 2], lw=1, label='W23')
    ax.plot(x, sg['W'][0][:, 2][:, 0], lw=1, label='W31')
    ax.plot(x, sg['W'][0][:, 2][:, 1], lw=1, label='W32')
    ax.plot(x, sg['W'][0][:, 2][:, 2], lw=1, label='W33')
    ax.plot(x, sg['W'][0][:, 3][:, 0], lw=1, label='W41')
    ax.plot(x, sg['W'][0][:, 3][:, 1], lw=1, label='W42')
    ax.plot(x, sg['W'][0][:, 3][:, 2], lw=1, label='W43')

    plt.legend()
    plt.show()


def spike_times_for_multiple_ng(ngs: list[NeuronGroup]):
    plt.rcParams.update({'font.size': 6})
    fig, ax = plt.subplots(figsize=(4, 3))

    for ind, ng in enumerate(ngs):
        ax.scatter(ng['spike'][0][ng['spike', 0][:, 1] == 1] * 0.01,
                   ng['spike'][0][ng['spike', 0][:, 1] == 1], s=1, label='ng%d' % (ind+1)),

    plt.xlabel('time', fontsize=12)
    plt.ylabel('iteration Nnumber', fontsize=12)
    plt.title('spike times', fontsize=12)
    plt.legend()
    plt.show()


def cosine_similarity_plot_v2(synapsGroups: list[SynapseGroup], m: int, n: int, y_lim: list = [-1, 1]) -> None:
    cosine_sims = []
    for i in range(690):
        x1 = torch.Tensor([])
        x2 = torch.Tensor([])
        for s in synapsGroups:
            x1 = torch.cat((x1, torch.Tensor(
                s['W'][0][:, 0][i, m], s['W'][0][:, 1][i, m], s['W'][0][:, 2][i, m], s['W'][0][:, 3][i, m])), 0)

            x2 = torch.cat((x1, torch.Tensor(
                s['W'][0][:, 0][i, n], s['W'][0][:, 1][i, n], s['W'][0][:, 2][i, n], s['W'][0][:, 3][i, n])), 0)

            cosine_sims.append(
                torch.nn.functional.cosine_similarity(x1, x2, dim=1))

    plt.rcParams.update({'font.size': 6})
    plt.figure(figsize=(4, 3))
    plt.plot(np.arange(690), cosine_sims)
    plt.xlabel('Iteration Nnumber', fontsize=8)
    plt.ylim(y_lim[0], y_lim[1])
    plt.ylabel('Cosine Sim', fontsize=8)
    plt.title('Cosine Similarity Of Weights', fontsize=8)
    plt.show()


def cosine_similarity_plot(s1: SynapseGroup, s2: SynapseGroup, y_lim: list = [-1, 1]) -> None:
    cosine_sims = []
    for i in range(690):
        x1 = torch.Tensor([[s1['W'][0][:, 0][i, 0], s1['W'][0][:, 1][i, 0], s1['W'][0][:, 2][i, 0], s1['W'][0][:, 3]
                          [i, 0], s2['W'][0][:, 0][i, 0], s2['W'][0][:, 1][i, 0], s2['W'][0][:, 2][i, 0], s2['W'][0][:, 3][i, 0]]])
        # x2 = torch.Tensor([[s3['W'][0][:, 0][i, 0], s3['W'][0][:, 1][i, 0], s3['W'][0][:, 2][i, 0], s3['W'][0][:, 3]
        #                   [i, 0], s4['W'][0][:, 0][i, 0], s4['W'][0][:, 1][i, 0], s4['W'][0][:, 2][i, 0], s4['W'][0][:, 3][i, 0]]])
        x2 = torch.Tensor([[s1['W'][0][:, 0][i, 1], s1['W'][0][:, 1][i, 1], s1['W'][0][:, 2][i, 1], s1['W'][0][:, 3]
                          [i, 1], s2['W'][0][:, 0][i, 1], s2['W'][0][:, 1][i, 1], s2['W'][0][:, 2][i, 1], s2['W'][0][:, 3][i, 1]]])
        cosine_sims.append(
            torch.nn.functional.cosine_similarity(x1, x2, dim=1))

    plt.rcParams.update({'font.size': 6})
    plt.figure(figsize=(4, 3))
    plt.plot(np.arange(690), cosine_sims)
    plt.xlabel('Iteration Nnumber', fontsize=8)
    plt.ylim(y_lim[0], y_lim[1])
    plt.ylabel('Cosine Sim', fontsize=8)
    plt.title('Cosine Similarity Of Weights', fontsize=8)
    plt.show()
