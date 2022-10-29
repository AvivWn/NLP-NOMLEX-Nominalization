import abc
from enum import Enum
from functools import lru_cache

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

from yet_another_verb.arguments_extractor.extraction import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.utils.parsed_argument import get_argument_in_parsed_text
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText


class EncodingLevel(str, Enum):
	HEAD_IDX = "head-idx"


class ArgumentEncoder(abc.ABC):
	def __init__(self, parsed_text: ParsedText):
		self.parsed_text = parsed_text

	@abc.abstractmethod
	def encode(self, argument: ExtractedArgument) -> torch.Tensor:
		pass


class PretrainedArgumentEncoder(ArgumentEncoder, abc.ABC):
	def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, device: str, parsed_text: ParsedText):
		super().__init__(parsed_text)
		self.tokenizer = tokenizer
		self.model = model
		self.device = device

	def _encode_sentence(self, sentence: str) -> torch.Tensor:
		tokenized = self.tokenizer(sentence.split(), return_tensors="pt", is_split_into_words=True, add_special_tokens=True)
		tokenized = tokenized.to(self.device)

		with torch.no_grad():
			return self.model(**tokenized)[0][0].cpu()


class HeadIdxArgumentEncoder(PretrainedArgumentEncoder):
	@lru_cache(maxsize=None)
	def encode(self, argument: ExtractedArgument):
		arg_span = get_argument_in_parsed_text(argument, self.parsed_text)
		head_idx = arg_span.root.i
		return self._encode_sentence(self.parsed_text.tokenized_text)[head_idx].clone()


ENCODER_BY_LEVEL = {
	EncodingLevel.HEAD_IDX: HeadIdxArgumentEncoder
}
