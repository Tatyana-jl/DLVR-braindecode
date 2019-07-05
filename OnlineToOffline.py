import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D


class OfflineProcess(object):


    def __init__(self, model, data, weights, gradients, optimizer_state):

        self.model = model
        self.layer_names = list(dict(self.model.named_children()).keys())

        self.nsteps = len(data)
        self.data = data
        self.weights = weights
        self.gradients = gradients
        self.optimizer_state = optimizer_state

    def plot_gradients(self, layers_to_plot=[]):

        gradients_trace = {}
        for step in range(self.nsteps):
            grad_step = torch.load(self.gradients[step])

            if len(layers_to_plot) == 0:
                layers_to_plot = list(grad_step.keys())

            for item in list(grad_step.keys()):
                if (item in layers_to_plot) & ('weight' in item) & ('conv' in item) & (grad_step[item] is not None):
                    grad_step[item] = torch.squeeze(grad_step[item])
                    try:
                        gradients_trace[item][:, step] = grad_step[item].detach().cpu().numpy()
                    except KeyError:
                        gradients_trace[item] = np.empty((grad_step[item].shape[0], self.nsteps, *grad_step[item].shape[1:]))

        time = np.arange(self.nsteps)

        for layer in list(gradients_trace.keys()):
            print(layer)
            fig = plt.figure(figsize=(40, 40))
            filters_per_layer = gradients_trace[layer].shape[0]
            for nfilt, filter in enumerate(gradients_trace[layer]):
                ax = fig.add_subplot((filters_per_layer//5)+1, 5, nfilt+1, projection='3d')
                nbins = 'auto'
                for time_step in time:
                    try:
                        hist, bins = np.histogram(filter[time_step].reshape((-1)).astype(np.float32), bins=nbins)
                        xs = (bins[:-1] + bins[1:]) / 2
                        ax.bar(xs, hist, zs=time_step, zdir='y', color='b', ec='b', alpha=0.8)
                    except ValueError:
                        print('Incorrect values for filter {:d} in time step {:d}, '
                              'instability error is triggered'.format(nfilt, time_step))
                        break
                    ax.set_xlabel('Filter gradient')
                    ax.set_ylabel('Time step')
                    ax.set_title('filter number' + str(nfilt))
            fig.suptitle(layer, fontsize=32)
            plt.show()

    def plot_weights(self, layers_to_plot=[]):

        weights_trace = {}
        for step in range(self.nsteps):
            weights_step = torch.load(self.weights[step])

            if len(layers_to_plot) == 0:
                layers_to_plot = list(weights_step.keys())

            for item in list(weights_step.keys()):
                if (item in layers_to_plot) & ('weight' in item) & ('conv' in item) & (weights_step[item] is not None):
                    weights_step[item] = torch.squeeze(weights_step[item])
                    try:
                        weights_trace[item][:, step] = weights_step[item].detach().cpu().numpy()
                    except KeyError:
                        weights_trace[item] = np.empty((weights_step[item].shape[0], self.nsteps, *weights_step[item].shape[1:]))

        time = np.arange(self.nsteps)

        for layer in list(weights_trace.keys()):
            print(layer)
            fig = plt.figure(figsize=(40, 40))
            filters_per_layer = weights_trace[layer].shape[0]
            for nfilt, filter in enumerate(weights_trace[layer]):
                ax = fig.add_subplot((filters_per_layer//5)+1, 5, nfilt+1, projection='3d')
                nbins = 'auto'
                for time_step in time:
                    try:
                        hist, bins = np.histogram(filter[time_step].reshape((-1, 1)).astype(np.float32), bins=nbins)
                        xs = (bins[:-1] + bins[1:]) / 2
                        ax.bar(xs, hist, zs=time_step, zdir='y', color='b', ec='b', alpha=0.8)
                    except ValueError:
                        print('Incorrect values for filter {:d} in time step {:d}, '
                              'instability error is triggered'.format(nfilt, time_step))
                        break
                    ax.set_xlabel('Filter weights')
                    ax.set_ylabel('Time step')
                    ax.set_title('filter number' + str(nfilt))
            fig.suptitle(layer, fontsize=32)
            plt.show()



