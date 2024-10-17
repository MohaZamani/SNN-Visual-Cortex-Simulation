import numpy as np
from abc import abstractmethod, ABC
import skimage.measure


class BasePooling(ABC):
    def __init__(self, size: tuple) -> None:
        self.size = size

    @abstractmethod
    def __call__(self, im: np.ndarray) -> np.ndarray:
        pass


class MaxPooling(BasePooling):

    def __call__(self, im: np.ndarray) -> np.ndarray:
        return skimage.measure.block_reduce(image=im, block_size=self.size, func=np.max)


class MeanPooling(BasePooling):

    def __call__(self, im: np.ndarray) -> np.ndarray:
        return skimage.measure.block_reduce(image=im, block_size=self.size, func=np.mean)


class MinPooling(BasePooling):

    def __call__(self, im: np.ndarray) -> np.ndarray:
        return skimage.measure.block_reduce(image=im, block_size=self.size, func=np.min)


class AvgPooling(BasePooling):

    def __call__(self, im: np.ndarray) -> np.ndarray:
        return skimage.measure.block_reduce(image=im, block_size=self.size, func=np.average)
