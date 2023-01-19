import abc

import torch


class Encoder(abc.ABC):
	@property
	@abc.abstractmethod
	def name(self) -> str:
		pass

	@property
	@abc.abstractmethod
	def encoding_size(self):
		pass

	@abc.abstractmethod
	def encode(self, sentence: str) -> torch.Tensor:
		pass

	@abc.abstractmethod
	def encode_word_in_context(self, sentence: str, word_idx: int) -> torch.Tensor:
		pass

	def __call__(self, *args, **kwargs):
		return self.encode(*args, **kwargs)
