from enum import Enum
from typing import Optional

from typeguard import typechecked

from yet_another_verb.arguments_extractor.extraction import Extraction, ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.argument.argument_type import ArgumentTypes
from yet_another_verb.arguments_extractor.extraction.utils.indices import get_indices_as_range
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedWords
from yet_another_verb.arguments_extractor.extraction.representation.parsed_representation import \
	ParsedRepresentation


class MentionType(str, Enum):
	TEXT_BOUND = "TextBoundMention"
	EVENT = "EventMention"


class ParsedOdinMentionRepresentation(ParsedRepresentation):
	def __init__(
			self, document_id: str, sentence_id: int, in_document_prefix: int,
			arg_types: Optional[ArgumentTypes] = None, use_head_idx_only: bool = False):
		super().__init__(arg_types)
		self.document_id = document_id
		self.sentence_id = sentence_id
		self.in_document_prefix = in_document_prefix
		self.use_head_idx_only = use_head_idx_only

	def _as_mention(
			self, mention_type: MentionType, words: ParsedWords, predicate_idx: int,
			start_idx: int, end_idx: int, label: str = None
	) -> dict:
		mention_id = f"T:{self.document_id}.{self.sentence_id}.{predicate_idx}.{start_idx}"
		if mention_type == MentionType.EVENT:
			mention_id = f"R:{self.document_id}.{self.sentence_id}.{predicate_idx}"

		mention = {
			"type": mention_type,
			"id": mention_id,
			"text": words[start_idx: end_idx + 1].tokenized_text,
			"labels": [label] if label is not None else ["\xa0"],
			"sentence": self.sentence_id,
			"document": self.document_id,
			"event": self.in_document_prefix + predicate_idx,  # real index in document
			"eventPOS": words[predicate_idx].pos
		}

		if mention_type == MentionType.TEXT_BOUND:
			mention["tokenInterval"] = {
				"start": start_idx,
				"end": end_idx + 1
			}
		elif mention_type == MentionType.EVENT:
			mention["trigger"] = self._as_mention(MentionType.TEXT_BOUND, words, predicate_idx, predicate_idx, predicate_idx)
			mention["arguments"] = {}

		return mention

	@typechecked
	def _represent_argument(self, words: ParsedWords, predicate_idx: int, argument: ExtractedArgument) -> dict:
		if self.use_head_idx_only:
			start_idx = end_idx = argument.head_idx
		else:
			start_idx = argument.start_idx
			end_idx = argument.end_idx

		return self._as_mention(MentionType.TEXT_BOUND, words, predicate_idx, start_idx, end_idx)

	@typechecked
	def _represent_predicate(self, words: ParsedWords, predicate_idx: int) -> int:
		return predicate_idx

	def represent_single(self, extraction: Extraction) -> list:
		start_idx, end_idx = get_indices_as_range(extraction.arg_indices)
		predicate_idx = extraction.predicate_idx
		predicate = extraction.words[predicate_idx].text
		event_mention = self._as_mention(MentionType.EVENT, extraction.words, predicate_idx, start_idx, end_idx, predicate)
		event_mention["arguments"] = {}

		extraction_repr = [event_mention]
		for arg in extraction.args:
			if self.arg_types is None or arg.arg_type in self.arg_types:
				arg_mention = self._represent_argument(extraction.words, predicate_idx, arg)

				if arg.arg_tag not in event_mention["arguments"]:
					event_mention["arguments"][arg.arg_tag] = []

				event_mention["arguments"][arg.arg_tag] += [arg_mention]
				extraction_repr.append(arg_mention)

		return extraction_repr
