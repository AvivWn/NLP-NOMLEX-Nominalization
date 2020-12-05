from collections import defaultdict
from itertools import product

from .structural_matcher import StructuralMatcher
from ..extraction.predicate import Predicate
from ..extraction.extraction import Extraction
from noun_as_verb.lexicon_representation.lexical_subcat import LexicalSubcat
from noun_as_verb.lexicon_representation.lexicon import Lexicon


class NomlexMatcher(StructuralMatcher):
	lexicon: Lexicon

	def __init__(self, lexicon):
		super().__init__()
		self.lexicon = lexicon

	@staticmethod
	def _match_subcat(subcat: LexicalSubcat, candidates, predicate: Predicate):
		arg_types = subcat.get_args_types()
		args_per_candidate = defaultdict(list)

		for complement_type in arg_types:
			arg = subcat.get_arg(complement_type)
			found_match = False

			for candidate in candidates:
				matched_argument = arg.try_to_match(candidate, predicate.get_token())

				if matched_argument is not None:
					args_per_candidate[candidate.get_token()].append(matched_argument)
					found_match = True

			if not found_match and subcat.is_required(complement_type):
				return {}

		return args_per_candidate

	@staticmethod
	def _combine_args(subcat: LexicalSubcat, args_per_candidate, predicate: Predicate):
		candidates = args_per_candidate.keys()
		matches = [dict(zip(candidates, args)) for args in product(*args_per_candidate.values())]
		extractions = []

		for match in matches:
			extraction = Extraction(subcat, list(match.values()))

			# Check constraints on the current extraction
			if subcat.check_constraints(extraction, predicate.get_token()):
				extractions.append(extraction)

		return extractions

	@staticmethod
	def _find_subcat_structures(subcat: LexicalSubcat, candidates, predicate: Predicate):
		args_per_candidate = NomlexMatcher._match_subcat(subcat, candidates, predicate)
		return NomlexMatcher._combine_args(subcat, args_per_candidate, predicate)

	def _search_nomlex_structures(self, predicate: Predicate):
		predicate_token = predicate.get_token()
		word_entries = self.lexicon.find(predicate_token)

		subcats = []
		for word_entry in word_entries:
			subcats += word_entry.get_subcats()

		return subcats

	def find_structures(self, candidates, predicate: Predicate):
		subcats = self._search_nomlex_structures(predicate)

		if not subcats:
			return []

		extractions = []
		for subcat in subcats:
			extractions += self._find_subcat_structures(subcat, candidates, predicate)

		return extractions
