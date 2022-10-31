import abc

import torch


class Encoder(abc.ABC):
	@property
	@abc.abstractmethod
	def name(self) -> str: pass

	@abc.abstractmethod
	def encode(self, sentence: str) -> torch.Tensor:
		pass

	def __call__(self, *args, **kwargs):
		return self.encode(*args, **kwargs)
