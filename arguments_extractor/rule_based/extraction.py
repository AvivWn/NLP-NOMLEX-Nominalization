from collections import defaultdict

from arguments_extractor.rule_based.extracted_argument import ExtractedArgument
from arguments_extractor.constants.lexicon_constants import *

class Extraction:
	match = defaultdict(ExtractedArgument)

	def __init__(self, subcat, arguments: list):
		self.subcat = subcat
		self.match = defaultdict(ExtractedArgument)

		for argument in arguments:
			if argument is None:
				continue

			complement_type = argument.get_real_complement_type()

			# Multiple PP should get different names (PP1 and PP2)
			# It is possible only for the default subcat
			if subcat.subcat_type == DEFAULT_SUBCAT and complement_type in self.match.keys() and complement_type == COMP_PP:
				self.match[COMP_PP].argument_name = COMP_PP1
				self.match[COMP_PP1] = self.match.pop(COMP_PP)

				argument.argument_name = COMP_PP2
				complement_type = COMP_PP2

			self.match[complement_type] = argument

	def get_complements(self):
		return list(self.match.keys())

	def get_argument(self, complement_type: str):
		return self.match[complement_type]

	def get_tokens(self):
		return [argument.argument_token for argument in self.match.values()]

	def get_arguments_idxs(self):
		return sorted([argument.get_argument_idx() for argument in self.match.values()])

	def add_argument(self, argument: ExtractedArgument):
		self.match[argument.get_real_complement_type()] = argument



	def is_sub_extraction(self, other_extraction):
		# Is the given extraction is a sub-extraction of this extraction?
		return self.as_span_dict().items() <= other_extraction.as_span_dict().items()

	def is_more_informative(self, other_extraction):
		# Is the given extraction is more informative than this extraction?

		# An extraction can be more informative than the other only if they include the same exact argument tokens
		if self.get_arguments_idxs() != other_extraction.get_arguments_idxs():
			return False

		other_reversed_idxs_match = dict([(argument.get_argument_idx(), complement_type) for complement_type, argument in other_extraction.match.items()])

		# Check for candidates with difference in their complement types and matched position
		for argument in self.match.values():
			other_complement_type = other_reversed_idxs_match[argument.get_argument_idx()]
			other_argument = other_extraction.match[other_complement_type]

			if argument.is_more_informative(other_argument):
				return True

		return False



	def as_properties_dict(self):
		extraction_as_dict = {}

		# Save only the argument propeties in the dictionary
		for complement_type, argument in self.match.items():
			extraction_as_dict[argument.argument_name] = argument.get_properties()

		return extraction_as_dict

	def as_span_dict(self, trim_arguments=True):
		extraction_as_dict = {}
		extraction_token_indices = [argument.argument_token.i for argument in self.match.values()]

		# Cleans the resulted extraction, deletes duplicates between arguments and translates arguments into spans
		for complement_type, argument in self.match.items():
			extraction_as_dict[argument.argument_name] = argument.as_span(extraction_token_indices, trim_arguments)

		return extraction_as_dict