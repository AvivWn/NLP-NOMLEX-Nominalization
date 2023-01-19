from functools import lru_cache
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModel, BatchEncoding

from yet_another_verb.sentence_encoding.encoder import Encoder
from yet_another_verb.utils.print_utils import print_if_verbose


class TorchPretrainedEncoder(Encoder):
	def __init__(self, encoder_name: str, device: str = None, **kwargs):
		self.device = device
		if self.device is None:
			self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.encoder_name = encoder_name
		self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
		self.model = AutoModel.from_pretrained(encoder_name).to(self.device)
		self.model.eval()

	@property
	def name(self) -> str:
		return self.encoder_name

	@property
	def encoding_size(self):
		return self.model.config.hidden_size

	def _tokenize(self, sentence: str):
		return self.tokenizer(
			sentence.split(), return_tensors="pt", is_split_into_words=True,
			add_special_tokens=True)

	@lru_cache(maxsize=None)
	def _tokenize_and_encode(self, sentence: str) -> Tuple[BatchEncoding, torch.Tensor]:
		tokenized = self._tokenize(sentence).to(self.device)

		with torch.no_grad():
			return tokenized, self.model(**tokenized)[0][0].cpu()

	def encode(self, sentence: str) -> torch.Tensor:
		return self._tokenize_and_encode(sentence)[1]

	def encode_word_in_context(self, sentence: str, word_idx: int) -> torch.Tensor:
		tokenized, sentence_encodings = self._tokenize_and_encode(sentence)

		matching_token_indices = tokenized.word_to_tokens(word_idx)
		if matching_token_indices is None:
			print_if_verbose("Couldn't find the matching token of the word", sentence, word_idx)
			matching_token_idx = word_idx
		else:
			matching_token_idx = matching_token_indices[0]

		return sentence_encodings[matching_token_idx]
