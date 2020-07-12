from collections import defaultdict

from arguments_extractor.rule_based.extracted_argument import ExtractedArgument
from arguments_extractor.constants.lexicon_constants import *

class Extraction:
	match = defaultdict(ExtractedArgument)

	def __init__(self, subcat, arguments: list):
		self.subcat = subcat
		self.match = defaultdict(ExtractedArgument)

		for argument in arguments:
			if argument is not None:
				self.match[argument.get_real_complement_type()] = argument

	def get_complements(self):
		return list(self.match.keys())

	def get_argument(self, complement_type: str):
		return self.match[complement_type]

	def add_argument(self, argument: ExtractedArgument):
		self.match[argument.get_real_complement_type()] = argument

	def get_arguments_idxs(self):
		return sorted([argument.get_argument_idx() for argument in self.match.values()])

	def is_sub_extraction(self, other_extraction):
		# Is the given extraction is a sub-extraction of this extraction?
		return self.as_dict().items() <= other_extraction.as_dict().items()

	def is_more_informative(self, other_extraction):
		# Is the given extraction is more informative than this extraction?

		# An extraction can be more informative than the other only if they include the same exact argument tokens
		if self.get_arguments_idxs() != other_extraction.get_arguments_idxs():
			return False

		other_reversed_idxs_match = dict([(argument.get_argument_idx(), complement_type) for complement_type, argument in other_extraction.match.items()])

		# Check for candidates with difference in their complement types and matched position
		for argument in self.match.values():
			complement_type = argument.get_real_complement_type()
			matched_position = argument.matched_position

			other_complement_type = other_reversed_idxs_match[argument.get_argument_idx()]
			other_argument = other_extraction.match[other_complement_type]
			other_matched_position = other_argument.matched_position

			if matched_position.islower() and other_matched_position.islower():
				# We should take the prefix argument with the most informative prefix (meaning "about what" and not just "about")
				if len(matched_position) < len(other_matched_position):
					return True

				# Simple preposition argument should not appear if there is another type of complement to the same argument
				if complement_type in [COMP_PP, COMP_PP1, COMP_PP2] and other_complement_type.startswith("P-"):
					return True

			# FOR-TO-INF is preferable than a standard TO-INF phrase
			if complement_type.startswith(COMP_TO_INF) and other_complement_type.startswith("FOR-"):
				return True

			# A complex prepositional phrase is preferable than a standard gerund phrase
			if matched_position == POS_ING and other_matched_position.islower():
				return True

			# NP VS PP- which one is more important? Both can be possible, os both will be extracted
			# if complement_type == COMP_PP and other_complement_type in [COMP_OBJ, COMP_IND_OBJ, COMP_SUBJ]:
			# 	return True

		return False

	def as_dict(self):
		extraction_as_dict = {}
		extraction_token_indices = [argument.argument_token.i for argument in self.match.values()]

		# Cleans the resulted extraction, deletes duplicates between arguments and translates arguments into spans
		for complement_type, argument in self.match.items():
			extraction_as_dict[argument.argument_name] = argument.get_argument_span(extraction_token_indices)

		return extraction_as_dict