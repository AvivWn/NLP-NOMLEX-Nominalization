from itertools import chain, permutations, product
from typing import List, Optional, Iterable, Union
from copy import deepcopy

from yet_another_verb.arguments_extractor.args_extractor import ArgsExtractor
from yet_another_verb.arguments_extractor.extraction import Extraction, Extractions, ArgumentType, ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.utils.filters import prefer_by_n_args, uniqify, \
	prefer_by_constraints
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.input_text import InputText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.exceptions import EmptyArgumentException
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.nomlex.nomlex_maestro import NomlexMaestro
from yet_another_verb.nomlex.nomlex_version import NomlexVersion
from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap
from yet_another_verb.configuration import EXTRACTORS_CONFIG
from yet_another_verb.utils.debug_utils import timeit
from yet_another_verb.utils.print_utils import print_as_title_if_verbose
from yet_another_verb.utils.hashing_utils import consistent_hash


class NomlexArgsExtractor(ArgsExtractor):
	def __init__(
			self,
			nomlex_version: NomlexVersion = EXTRACTORS_CONFIG.NOMLEX_VERSION,
			dependency_parser: DependencyParser = DependencyParserFactory()(),
			**kwargs
	):
		self.adapted_lexicon = timeit(NomlexMaestro(nomlex_version).get_adapted_lexicon)()
		self.dependency_parser = dependency_parser

	def _tokenize(self, text: InputText) -> ParsedText:
		return self.dependency_parser(text)

	def _is_potential_predicate(
			self, word_idx: int, words: list,
			limited_predicates: Optional[list], limited_postags: Optional[list],
			allow_related_forms: bool
	) -> bool:
		word = words[word_idx]

		if limited_postags is not None:
			if word.tag not in limited_postags and word.pos not in limited_postags:
				return False

		if limited_predicates is None:
			return True

		if allow_related_forms:
			self.adapted_lexicon.enhance_orths(limited_predicates)

		return word.lemma in limited_predicates

	def _is_potential_sentence(
			self, words: list,
			limited_predicates: Optional[list], allow_related_forms: bool
	) -> bool:
		if limited_predicates is None:
			return True

		lemmas = set([w.lemma for w in words])

		if allow_related_forms:
			self.adapted_lexicon.enhance_orths(limited_predicates)

		return not lemmas.isdisjoint(limited_predicates)

	@staticmethod
	def _is_not_empty_and_disjoint(l1: Union[list, set], l2: Union[list, set]):
		return len(l1) > 0 and len(l2) > 0 and set(l1).isdisjoint(l2)

	def _is_word_match_constraints(self, word: ParsedWord, constraints_map: ConstraintsMap) -> bool:
		required_contraints = [
			lambda: not self._is_not_empty_and_disjoint(constraints_map.values, {word.text}),
			lambda: not self._is_not_empty_and_disjoint(constraints_map.postags, {word.tag, word.pos}),
			lambda: not self._is_not_empty_and_disjoint(constraints_map.dep_relations, {word.dep})
		]

		return all(constraint() is True for constraint in required_contraints)

	@staticmethod
	def _filter_word_relatives(word: ParsedWord, constraints_map: ConstraintsMap) -> List[ParsedWord]:
		relevant_relations, relevant_postags, relevant_values = set(), set(), set()
		for m in constraints_map.relatives_constraints:
			relevant_relations.update(m.dep_relations)
			relevant_postags.update(m.postags)
			relevant_values.update(m.values)

			# is JOKER
			if len(m.postags) == 0 and len(m.dep_relations) == 0 and len(m.values) == 0:
				return [relative for relative in word.children]

		relevant_checkers = [
			lambda relative: relative.dep in relevant_relations,
			lambda relative: relative.tag in relevant_postags or relative.pos in relevant_postags,
			lambda relative: relative.text in relevant_values
		]

		return [relative for relative in word.children if any(c(relative) for c in relevant_checkers)]

	def _get_relatives_matched_args(
			self, predicate: ParsedWord, relatives: Iterable[Optional[ParsedWord]],
			constraints_maps: List[Optional[ConstraintsMap]]
	) -> Optional[List[List[ExtractedArgument]]]:
		relatives_matched_args = []
		for i, relative in enumerate(relatives):
			if i >= len(constraints_maps):
				break

			constraints_map = constraints_maps[i]
			if constraints_map is None:
				continue

			relative_matched_args = None if relative is None else self._get_matched_arguments(
				predicate, relative, constraints_map)

			if relative_matched_args is None:
				if constraints_map.required:
					return None

				continue

			relatives_matched_args.append(relative_matched_args)

		return [list(chain(*combined_args)) for combined_args in product(*relatives_matched_args)]

	@staticmethod
	def _get_combined_args(args: List[ExtractedArgument], new_arg: ExtractedArgument, constraints_map: ConstraintsMap) -> List[ExtractedArgument]:
		combined_args_by_type = {new_arg.arg_type: new_arg}
		other_args_indices = set()
		typeless_args = []

		for other_arg in args:
			if other_arg.arg_type is None:
				typeless_args.append(other_arg)
			elif other_arg.arg_type not in combined_args_by_type:
				combined_args_by_type[other_arg.arg_type] = other_arg
			else:
				combined_args_by_type[other_arg.arg_type].add_indices(other_arg.arg_indices)
				combined_args_by_type[other_arg.arg_type].fulfilled_constraints += other_arg.fulfilled_constraints

			if other_arg.arg_type != constraints_map.arg_type:
				other_args_indices.update(other_arg.arg_indices)

		new_arg.remove_indices(other_args_indices)
		return list(combined_args_by_type.values()) + typeless_args

	def _has_matching_potential(self, relatives: List[ParsedWord], constraints_map: ConstraintsMap) -> bool:
		required_constraints = constraints_map.get_required_relative_maps()
		if len(relatives) < len(required_constraints):
			return False

		existing_relations, existing_postags, existing_values = set(), set(), set()
		for word in relatives:
			existing_relations.add(word.dep)
			existing_postags.update([word.tag, word.pos])
			existing_values.add(word.text)

		required_constraints_checkers = [
			lambda x: self._is_not_empty_and_disjoint(x.values, existing_values),
			lambda x: self._is_not_empty_and_disjoint(x.dep_relations, existing_relations),
			lambda x: self._is_not_empty_and_disjoint(x.postags, existing_postags)
		]

		if any(any(checker(x) for checker in required_constraints_checkers) for x in required_constraints):
			return False

		return True

	def _get_matched_arguments(
			self, predicate: ParsedWord, word: ParsedWord, constraints_map: ConstraintsMap
	) -> Optional[List[List[ExtractedArgument]]]:
		is_match_constraints = self._is_word_match_constraints(word, constraints_map)
		if not is_match_constraints:
			return None

		required_constraints = constraints_map.get_required_relative_maps()
		if len(word.children) < len(required_constraints):
			return None

		relevant_relatives = self._filter_word_relatives(word, constraints_map)
		if not self._has_matching_potential(relevant_relatives, constraints_map):
			return None

		matched_args_combinations = []
		relevant_relatives += [None] * (len(constraints_map.relatives_constraints) - len(relevant_relatives))
		relatives_permutations = set(permutations(relevant_relatives, len(constraints_map.relatives_constraints)))

		for relatives_permute in relatives_permutations:
			relatives_matched_args_combinations = self._get_relatives_matched_args(
				predicate=predicate, relatives=relatives_permute,
				constraints_maps=constraints_map.relatives_constraints)

			if relatives_matched_args_combinations is not None:
				matched_args_combinations += relatives_matched_args_combinations

		arg_indices = word.subtree_indices if predicate != word else []
		arg_indices = list(set(arg_indices + [word.i]))
		new_arg = ExtractedArgument(
			start_idx=min(arg_indices),
			end_idx=max(arg_indices),
			arg_type=constraints_map.arg_type if is_match_constraints else None,
			fulfilled_constraints=[constraints_map] if is_match_constraints else []
		)

		for i, matched_args in enumerate(matched_args_combinations):
			matched_args_combinations[i] = self._get_combined_args(deepcopy(matched_args), deepcopy(new_arg), constraints_map)

		return matched_args_combinations

	@staticmethod
	def _reorder_numbered_args(extracted_args: List[ExtractedArgument]):
		arg_by_type = {arg.arg_type: arg for arg in extracted_args}
		pp1_arg = arg_by_type.get(ArgumentType.PP1)
		pp2_arg = arg_by_type.get(ArgumentType.PP2)

		if pp1_arg is not None and pp2_arg is not None:
			if pp1_arg.start_idx > pp2_arg.start_idx:
				pp1_arg.arg_type = ArgumentType.PP2
				pp1_arg.arg_tag = ArgumentType.PP2
				pp2_arg.arg_type = ArgumentType.PP1
				pp2_arg.arg_tag = ArgumentType.PP1

	def extract(self, word_idx: int, words: ParsedText) -> Optional[Extractions]:
		word = words[word_idx]
		word_entry = self.adapted_lexicon.entries.get(word.lemma, None)

		if word_entry is None:
			return []

		print_as_title_if_verbose(f"{word.text} ({word.lemma}, {word.pos})")
		relevant_constraints_maps = word_entry.get_relevant_constraints_maps(word=word, order_by_potential=True)

		extractions = []
		max_num_of_args = 0
		for i, constraints_map in enumerate(relevant_constraints_maps):
			if len(constraints_map.included_args) < max_num_of_args:
				break

			try:
				matched_args_combinations = self._get_matched_arguments(word, word, constraints_map)
			except EmptyArgumentException:
				# Might happen in some dependency parsing trees due to complex dependency relations, but ignored due to rarity
				continue

			if matched_args_combinations is None:
				continue

			for matched_args in matched_args_combinations:
				self._reorder_numbered_args(matched_args)

				if any(arg.arg_type is not None for arg in matched_args):
					extraction = Extraction(
						words=words, predicate_idx=word_idx, predicate_lemma=word.lemma,
						args=list(set(matched_args)),
						fulfilled_constraints=constraints_map
					)
					extractions.append(extraction)
					max_num_of_args = max(max_num_of_args, len(extraction))

		# Order extractions by constraints (for consistency)
		extractions = sorted(extractions, key=lambda e: consistent_hash(
			str(e.fulfilled_constraints) + str(e.args) + str(e.typeless_args)))

		for filter_func in [uniqify, prefer_by_n_args, prefer_by_constraints]:
			extractions = filter_func(extractions)

		return extractions
