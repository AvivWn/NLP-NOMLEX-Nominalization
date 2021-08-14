from itertools import chain
from typing import List, Optional

from yet_another_verb.arguments_extractor.args_extractor import ArgsExtractor
from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.extraction import Extraction
from yet_another_verb.arguments_extractor.extraction.filters import choose_longest, uniqify
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.input_text import InputText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.nomlex.nomlex_maestro import NomlexMaestro
from yet_another_verb.nomlex.nomlex_version import NomlexVersion
from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap
from yet_another_verb.configuration import NOMLEX_CONFIG, PARSING_CONFIG


class NomlexArgsExtractor(ArgsExtractor):
	def __init__(
			self,
			nomlex_version: NomlexVersion = NOMLEX_CONFIG.NOMLEX_VERSION,
			dependency_parser: DependencyParser = PARSING_CONFIG.DEFAULT_PARSER_MAKER()
	):
		self.adapted_lexicon = NomlexMaestro(nomlex_version).get_adapted_lexicon()
		self.dependency_parser = dependency_parser

	def preprocess(self, text: InputText) -> ParsedText:
		return self.dependency_parser(text)

	@staticmethod
	def _is_empty_or_contain(values: list, v):
		return len(values) == 0 or v in values

	def _is_token_match_constraints(self, token: ParsedWord, constraint_map: ConstraintsMap) -> bool:
		required_contraints = [
			lambda: self._is_empty_or_contain(constraint_map.values, token.text),
			lambda: self._is_empty_or_contain(constraint_map.postags, token.tag),
			lambda: self._is_empty_or_contain(constraint_map.word_relations, token.dep)
		]

		return all(constraint() is True for constraint in required_contraints)

	def _get_matched_arguments(
			self, token: ParsedWord, constraint_map: ConstraintsMap
	) -> Optional[List[ExtractedArgument]]:
		arg_idxs = []
		if not self._is_token_match_constraints(token, constraint_map):
			if constraint_map.required:
				return None

		elif constraint_map.arg_type is not None:
			arg_idxs = [token.i] if constraint_map.word_relations == [] else token.subtree_indices

		extracted_args = []
		for sub_contraint in constraint_map.sub_constraints:
			sub_extracted_args = self._get_matched_arguments(token, sub_contraint)
			if sub_extracted_args is None:
				return None

			extracted_args += sub_extracted_args

		other_args_idxs = chain(*[arg.arg_idxs for arg in extracted_args])
		arg_idxs = [i for i in arg_idxs if i not in other_args_idxs]

		if len(arg_idxs) > 0:
			extracted_args.append(ExtractedArgument(
				arg_idxs=arg_idxs,
				arg_type=constraint_map.arg_type
			))

		return extracted_args

	def extract(self, word_idx: int, parsed_text: ParsedText) -> List[Extraction]:
		word = parsed_text[word_idx]
		word_entry = self.adapted_lexicon.entries.get(word.lemma)

		if word_entry is None:
			return []

		maps = list(chain(*word_entry.subcats.values()))

		extractions = []
		for constraint_map in maps:
			for token in word.children:
				matched_args = self._get_matched_arguments(token, constraint_map)
				if matched_args is not None:
					extractions.append(Extraction(
						predicate_idx=word_idx,
						args=matched_args
					))

		return choose_longest(uniqify(extractions))
