import fasttext
import torch

from yet_another_verb.sentence_encoding.encoder import Encoder


class FastTextEncoder(Encoder):
	def __init__(self, encoder_name: str, **kwargs):
		self.encoder_name = encoder_name
		self.model = fasttext.load_model(encoder_name)

	@property
	def name(self) -> str:
		return self.encoder_name

	@property
	def encoding_size(self):
		return self.model.get_dimension()

	def encode(self, sentence: str) -> torch.Tensor:
		return torch.tensor(self.model.get_sentence_vector(sentence))

	def encode_word_in_context(self, sentence: str, word_idx: int) -> torch.Tensor:
		raise NotImplementedError()
