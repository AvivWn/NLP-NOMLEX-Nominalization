from collections import defaultdict

from noun_as_verb.rule_based import ExtractedArgument
from noun_as_verb.constants.lexicon_constants import *

class Extraction:
	match = defaultdict(ExtractedArgument)

	def __init__(self, subcat, arguments: list):
		self.subcat = subcat
		self.match = defaultdict(ExtractedArgument)

		for argument in arguments:
			if argument is None:
				continue

			complement_type = argument.get_real_type()

			# Multiple PP should get different names (PP1 and PP2)
			# It is possible only for the default subcat
			if subcat.is_default_subcat() and complement_type == COMP_PP in self.match.keys():
				self.match[COMP_PP].set_name(COMP_PP1)
				self.match[COMP_PP1] = self.match.pop(COMP_PP)

				argument.set_name(COMP_PP2)
				complement_type = COMP_PP2

			self.match[complement_type] = argument

	def __eq__(self, other):
		return self.as_properties_dict() == other.as_properties_dict()


	def get_complements(self):
		return list(self.match.keys())

	def get_argument(self, complement_type: str):
		return self.match[complement_type]

	def get_arguments(self):
		return self.match.values()

	def get_tokens(self):
		return [argument.get_token() for argument in self.match.values()]

	def get_arguments_idxs(self):
		return sorted([argument.get_head_idx() for argument in self.match.values()])

	def get_match(self):
		return self.match

	# def get_filtered(self, candidates_args: dict):
	# 	# Returns the filtered extraction based on the given dictionary
	# 	filtered_match = {k:arg for k,arg in self.match.items()
	# 					  if arg in candidates_args[arg.get_token()]}
	#
	# 	self.match = filtered_match
	# 	return self

		#filtered = Extraction(self.subcat, [])
		#iltered.match = filtered_match
		#return filtered

	def get_filtered(self, candidate2type: dict):
		# Returns the filtered extraction based on the given dictionary
		filtered_match = {role_type:arg for role_type, arg in self.match.items()
						  if arg.get_span() in candidate2type and role_type in candidate2type[arg.get_span()]}

		self.match = filtered_match
		return self

		#filtered = Extraction(self.subcat, [])
		#iltered.match = filtered_match
		#return filtered

	def get_candidates_args(self):
		# Returns a dictionary of all the possible arguments for each candidate
		candidates_args = defaultdict(list)
		for argument in self.match.values():
			candidate = argument.get_token()
			candidates_args[candidate].append(argument)

		return candidates_args

	def get_candidates_types(self):
		# Returns a dictionary of all the possible arguments for each candidate
		candidate2types = defaultdict(list)
		for argument in self.match.values():
			candidate = argument.as_span(trim_all=True)

			if candidate:
				candidate2types[candidate].append(argument.get_real_type())

		return candidate2types

	def add_argument(self, argument: ExtractedArgument):
		self.match[argument.get_real_type()] = argument



	def isin(self, extractions):
		return any(self == e for e in extractions)

	def is_sub_extraction(self, other_extraction):
		# Is the given extraction is a sub-extraction of this extraction?

		span_dict = self.as_span_dict()
		other_span_dict = other_extraction.as_span_dict()

		# Ignore PP1 and PP2 if they match each other
		include_two_pp = lambda d: {COMP_PP1, COMP_PP2}.issubset(d.keys())
		if include_two_pp(span_dict) and include_two_pp(other_span_dict):
			if span_dict[COMP_PP1] == other_span_dict[COMP_PP2] and span_dict[COMP_PP2] == other_span_dict[COMP_PP1]:
				span_dict.pop(COMP_PP1)
				span_dict.pop(COMP_PP2)
				other_span_dict.pop(COMP_PP1)
				other_span_dict.pop(COMP_PP2)

		return span_dict.items() <= other_span_dict.items()

	def is_more_informative(self, other_extraction):
		# Is the given extraction is more informative than this extraction?

		# An extraction can be more informative than the other only if they include the same exact argument tokens
		if self.get_arguments_idxs() != other_extraction.get_arguments_idxs():
			return False

		other_reversed_idxs_match = dict([(argument.get_head_idx(), complement_type) for complement_type, argument in other_extraction.match.items()])

		# Check for candidates with difference in their complement types and matched position
		for argument in self.match.values():
			other_complement_type = other_reversed_idxs_match[argument.get_head_idx()]
			other_argument = other_extraction.match[other_complement_type]

			if argument.is_more_informative(other_argument):
				return True

		return False



	def as_properties_dict(self):
		as_dict = {}

		# Save only the argument propeties in the dictionary
		for complement_type, arg in self.match.items():
			as_dict[arg.get_name()] = arg.get_properties()

		return as_dict

	def as_span_dict(self, trim_arguments=True):
		as_dict = {}

		# Cleans the extraction, deletes duplicates between args and translates args into spans
		for arg in self.match.values():
			arg_span = arg.as_span(trim_arguments)

			if arg_span:
				as_dict[arg.get_name()] = arg_span

		return as_dict