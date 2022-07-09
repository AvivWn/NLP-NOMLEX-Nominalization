from collections import defaultdict, namedtuple
from itertools import product
from typing import List, TYPE_CHECKING, Iterator, Optional, Set, Dict

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord
from yet_another_verb.utils.print_utils import print_if_verbose

if TYPE_CHECKING:
	from yet_another_verb.nomlex.representation.constraints_map import ConstraintsMap

WordProperty = namedtuple("WordProperty", ["value", "postag", "relation"])


@dataclass_json
@dataclass
class IndexedConstraintsMaps:
	constraints_maps: List['ConstraintsMap'] = field(default_factory=list)

	@staticmethod
	def _expand_with_suitable_properties(
			property_mapping: Dict[WordProperty, Set[int]],
			constraints_map: 'ConstraintsMap',
			map_idx: int):
		values = constraints_map.values if len(constraints_map.values) > 0 else [None]
		postags = constraints_map.postags if len(constraints_map.postags) > 0 else [None]
		relations = constraints_map.word_relations if len(constraints_map.word_relations) > 0 else [None]

		for value, postag, relation in product(values, postags, relations):
			word_property = WordProperty(value=value, postag=postag, relation=relation)
			property_mapping[word_property].add(map_idx)

	def __post_init__(self):
		self._ordered_idxs = list(range(len(self.constraints_maps)))
		self._ordered_idxs.sort(key=lambda c_id: len(self.constraints_maps[c_id].included_args), reverse=True)

		self._required_idxs = set()
		self._all_optional_relative_idxs = set()
		self._idxs_by_properties = defaultdict(set)
		self._idxs_by_relative_properties = defaultdict(set)

		for i, constraints_map in enumerate(self.constraints_maps):
			if constraints_map.required:
				self._required_idxs.add(i)

			required_relative_maps = constraints_map.get_required_relative_maps()
			if len(required_relative_maps) == 0:
				self._all_optional_relative_idxs.add(i)

			self._expand_with_suitable_properties(self._idxs_by_properties, constraints_map, i)

			for relative_map in constraints_map.relatives_constraints:
				self._expand_with_suitable_properties(self._idxs_by_relative_properties, relative_map, i)

	@staticmethod
	def _get_relevants_idxs_in_mapping(property_mapping: Dict[WordProperty, Set[int]], word: ParsedWord) -> Set[int]:
		relevant_idxs = set()
		for value, postag, relation in product([word.text, None], [word.tag, None], [word.dep, None]):
			word_property = WordProperty(value=value, postag=postag, relation=relation)
			relevant_idxs.update(property_mapping[word_property])

		return relevant_idxs

	def _reorder_by_potential(self, idxs: Set[int]) -> List[int]:
		ordered_idxs = []
		for idx in self._ordered_idxs:
			if idx in idxs:
				ordered_idxs.append(idx)

		return ordered_idxs

	def _get_relevant_constraints_maps_idxs(
			self, word: ParsedWord, relatives: Optional[List[ParsedWord]] = None,
			order_by_potential: bool = False
	) -> List[int]:
		relevant_idxs = self._get_relevants_idxs_in_mapping(self._idxs_by_properties, word)

		if relatives is not None:
			relatives_relevant_idxs = set()
			for relative in relatives:
				relatives_relevant_idxs.update(
					self._get_relevants_idxs_in_mapping(self._idxs_by_relative_properties, relative))

			relatives_relevant_idxs.update(self._all_optional_relative_idxs)
			relevant_idxs = relevant_idxs.intersection(relatives_relevant_idxs)

		if order_by_potential:
			relevant_idxs = self._reorder_by_potential(relevant_idxs)
		else:
			relevant_idxs = list(relevant_idxs)

		print_if_verbose("SAVED:", len(self.constraints_maps) - len(relevant_idxs))
		return relevant_idxs

	def get_relevant_constraints_maps(
			self, word: ParsedWord, order_by_potential: bool = False) -> Iterator['ConstraintsMap']:
		relevant_idxs = self._get_relevant_constraints_maps_idxs(word, word.children, order_by_potential)
		return map(lambda i: self.constraints_maps[i], relevant_idxs)

	def get_relevant_maps_combinations(self, words: List[ParsedWord]) -> Iterator[List[Optional['ConstraintsMap']]]:
		coresponding_relevant_maps_idxs = []
		for i, word in enumerate(words):
			relevant_maps_idxs = self._get_relevant_constraints_maps_idxs(word)
			coresponding_relevant_maps_idxs.append(relevant_maps_idxs + [-i])

		maps_idxs_permutations = product(*coresponding_relevant_maps_idxs)

		for idxs_permute in maps_idxs_permutations:
			if len(set(idxs_permute)) < len(idxs_permute):
				continue

			if not self._required_idxs.issubset(idxs_permute):
				continue

			yield [self.constraints_maps[i] if i >= 0 else None for i in idxs_permute]
