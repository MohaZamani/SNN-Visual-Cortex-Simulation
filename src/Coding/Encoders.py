import torch
import numpy as np
import tensorflow as tf
from abc import abstractmethod, ABC


class Encoding(ABC):
    def __init__(self, time: float, dt: int = 1) -> None:
        self.time = time
        self.dt = dt

    @abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        pass


class TTFSCoding(Encoding):
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        interval_width = 1.0 / self.time

        output = torch.zeros((len(data), self.time))

        for i, val in enumerate(data):
            interval_index = self.time - 1 - max(0, min(
                int(val / interval_width) - 1, (self.time) - 1))

            output[i, interval_index] = 1

        return output


class NeumericalCoding(Encoding):
    def __normal_dist(self, x, mean, sd):
        prob_density = (sd * torch.tensor(np.pi)
                        ).mul(torch.exp(-0.5 * ((x - mean) / sd) ** 2))
        return torch.round(prob_density).int()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = data * 256

        std = self.time / (torch.pi ** 3)

        output = torch.zeros((len(data), self.time))

        for i, val in enumerate(data):
            mean_values = torch.arange(0, 256, 256 / self.time)
            prob_densities = self.__normal_dist(val, mean_values, std)

            # Get non-zero indices
            indices = torch.nonzero(prob_densities).squeeze(-1)
            indices = indices.clamp(max=self.time-1)

            if len(indices) > 0:
                output[i, self.time - indices - 1] = 1

        return output


class PoissonCoding(Encoding):
    def __init__(self, time_window, ratio):
        self.time_window = time_window
        self.ratio = ratio

    def __call__(self, img):
        if type(img) is tuple:
            return tuple([self(sub_inp) for sub_inp in img])

        original_shape, original_size = img.shape, img.numel()
        flat_img = img.view((-1,)) * self.ratio
        non_zero_mask = flat_img != 0

        flat_img[non_zero_mask] = 1 / flat_img[non_zero_mask]

        dist = torch.distributions.Poisson(rate=flat_img, validate_args=False)
        intervals = dist.sample(sample_shape=torch.Size([self.time_window]))
        intervals[:,
                  non_zero_mask] += (intervals[:, non_zero_mask] == 0).float()

        times = torch.cumsum(intervals, dim=0).long()
        times[times >= self.time_window + 1] = 0

        spike = torch.zeros(
            self.time_window + 1, original_size, device=img.device, dtype=torch.int
        )
        spike[times, torch.arange(original_size, device=img.device)] = 1
        spike = spike[1:]

        return spike.view(*original_shape, self.time_window)
