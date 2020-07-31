from itertools import product
from collections import defaultdict

from spacy.tokens import Token

from arguments_extractor.rule_based.lexical_argument import LexicalArgument
from arguments_extractor.rule_based.extraction import Extraction
from arguments_extractor.constants.lexicon_constants import *
from arguments_extractor.rule_based.utils import get_argument_candidates
from arguments_extractor.utils import difference_list, get_linked_arg

class LexicalSubcat:
	subcat_type: str

	arguments = defaultdict(LexicalArgument)
	requires: list
	optionals: list
	constraints: list
	not_constraints: dict
	is_verb: bool

	def __init__(self, subcat_info: dict, subcat_type, is_verb):
		self.subcat_type = subcat_type

		self.arguments = defaultdict(LexicalArgument)
		for complement_type in difference_list(subcat_info.keys(), [SUBCAT_REQUIRED, SUBCAT_OPTIONAL, SUBCAT_NOT, SUBCAT_CONSTRAINTS]):
			self.arguments[complement_type] = LexicalArgument(subcat_info[complement_type], complement_type, is_verb)

		self.requires = subcat_info.get(SUBCAT_REQUIRED, [])
		self.optionals = difference_list(self.arguments.keys(), subcat_info.get(SUBCAT_REQUIRED, []))
		self.not_constraints = subcat_info.get(SUBCAT_NOT, [])
		self.constraints = subcat_info.get(SUBCAT_CONSTRAINTS, [])
		self.is_verb = is_verb

	# Subcat constraints

	def _check_ordering(self, match: dict, referenced_token: Token):
		"""
		Checks that the order of the pre-nom arguments is as "SUBJECT > INDIRECT OBJECT > DIRECT OBJECT > OBLIQUE" ("Ordering Constraint", NOMLEX manual)
		This constraint is relevant only to nominalizations
		:param match: the match between the arguments types and the actual arguments ({COMP: Token})
		:param referenced_token: the predicate of the arguments that we are after
		:return: True if the given match is compatible with the "Ordering Constraint", and False otherwise
		"""

		if self.is_verb:
			return True

		subject_index = match.get(COMP_SUBJ, referenced_token).i
		ind_object_index = match.get(COMP_IND_OBJ, referenced_token).i
		object_index = match.get(COMP_OBJ, referenced_token).i
		other_indexes = [referenced_token.i]

		for complement_type, complement_token in match.items():
			if complement_type not in [COMP_SUBJ, COMP_OBJ, COMP_IND_OBJ]:
				other_indexes.append(complement_token.i)

		# Subject must appear before ind-object and object and others, if it appears before nom
		if subject_index < referenced_token.i and subject_index > min([ind_object_index, object_index] + other_indexes):
			return False

		# Ind-object must appear before object and any others, if it appears before nom
		if ind_object_index < referenced_token.i and ind_object_index > min([object_index] + other_indexes):
			return False

		# Object must appear before any others, if it appears before nom
		if object_index < referenced_token.i and object_index > min(other_indexes):
			return False

		return True

	def _check_uniqueness(self, complement_types: list, matched_positions: list):
		"""
		Checks that different arguments of noms don't appear using the same positions (relevant to specific positions)
		And that any complement doesn't appear more than once
		:param complement_types: a list of complement types
		:param matched_positions: a list of the corresponding matched positions for the complements
		:return: True if the given match is compatible with the "Uniqueness Principle", and False otherwise
		"""

		# Any complement can appear at most once
		if len(complement_types) != len(set(complement_types)):
			return False

		if self.is_verb:
			return True

		# The next positions can appear at most once (for nominalizations)
		for position in [POS_DET_POSS, "of"]:
			if list(matched_positions).count(position) > 1:
				return False

		return True

	def _is_in_not(self, complement_types: list, matched_positions: list):
		"""
		Checks whether the given match between arguments and positions appear in the NOT constraints
		:param complement_types: a list of complement types
		:param matched_positions: a list of the corresponding matched positions for the complements
		:return: True if this match don't appear in the NOT constraints of this subcat, and False otherwise
		"""

		if self.not_constraints == []:
			return False

		for not_constraint in self.not_constraints:
			# The NOT constraint isn't relevant it doesn't intersect with the complement types
			if difference_list(complement_types, not_constraint.keys()) == complement_types:
				continue

			# Check for any violation with the current NOT constraint
			found_violation = False
			for complement_type, matched_position in zip(complement_types, matched_positions):
				if complement_type in not_constraint.keys() and \
						matched_position not in not_constraint[complement_type]:
					found_violation = True
					break

			if found_violation:
				return False

		return True

	def _check_if_no_other_object(self, complement_types: list, matched_positions: list):
		"""
		Checks the compatibility of a single object argument of nominalizations (not relevant to verbs)
		This function has any meaning only when there is signle object argument
		:param complement_types: a list of complement types
		:param matched_positions: a list of the matched positions for the complements
		:return: True if a single object (if any) gets an appropriate position
		"""

		if self.is_verb:
			return True

		object_args = []

		# Get all the "objects" that are arguments of the nominalization
		for object_candidate in [COMP_OBJ, COMP_IND_OBJ, COMP_SUBJ]:
			if object_candidate in complement_types:
				object_args.append(object_candidate)

		if len(object_args) != 1:
			return True

		# Only one argument was founded
		complement_index = complement_types.index(object_args[0])
		complement_type = complement_types[complement_index]
		matched_position = matched_positions[complement_index]

		# Check if there isn't any other argument that should get that position when it is the only object
		# Meaning- this argument cannot get this position when it is the only object argument
		for other_complement_type in difference_list(self.arguments.keys(), [complement_type]):
			if matched_position == POS_DET_POSS:
				if self.arguments[other_complement_type].is_det_poss_only(get_linked_arg(self.is_verb)):
					return False

			elif matched_position == POS_N_N_MOD:
				if self.arguments[other_complement_type].is_n_n_mod_only(get_linked_arg(self.is_verb)):
					return False

		return True

	def _check_constraints(self, extraction: Extraction, referenced_token: Token):
		"""
		Checks whether the given subcat match is compatible with the constraints of this subcat
		:param extraction: an extraction object that includes the matches between the arguments types and the actual arguments
		:param referenced_token: the predicate of the arguments that we are after
		:return: True if the candidate doesn't contradict the constraints, and False otherwise
		"""

		if extraction.get_complements() == []:
			return False

		complement_types = extraction.get_complements()
		argument_tokens = [extraction.get_argument(complement_type).argument_token for complement_type in complement_types]
		matched_positions = [extraction.get_argument(complement_type).matched_position for complement_type in complement_types]
		match = dict(zip(complement_types, argument_tokens))

		####################################
		# Check subcat constraints regarding just the complement types

		# Check that all the required arguments appear in the given extraction
		if not set(self.requires).issubset(complement_types):
			return False

		# Check that the given extraction satisfied the "Ordering Constraint" (NOMLEX manual)
		if not self._check_ordering(match, referenced_token):
			return False

		# Check the boolean constraints
		if SUBCAT_CONSTRAINT_ADVP_OR_ADJP in self.constraints:
			if COMP_ADVP not in complement_types and COMP_ADJP not in complement_types:
				return False

		####################################
		# Check subcat constraints regarding the complement types and their matched positions

		# The sbuject of a *verb* may follow "by" only if the verb isn't written in passive voice
		if self.is_verb and COMP_SUBJ in complement_types:
			if extraction.match[COMP_SUBJ].matched_position == "by" and \
				(COMP_OBJ not in complement_types or extraction.match[COMP_OBJ].matched_position != POS_NSUBJPASS):
				return False

		# Check that the arguments and positions not appear in the NOT position constraints
		if self._is_in_not(complement_types, matched_positions):
			return False

		# Check that the arguments and positions satisfied the "Uniqueness Principle" (NOMLEX manual)
		if not self._check_uniqueness(complement_types, matched_positions):
			return False

		# Check that the position of a single object (only if it single) is legitimate
		if not self._check_if_no_other_object(complement_types, matched_positions):
			return False

		# Check that all the plurality constraint is being satisfied
		for complement_type, extracted_argument in extraction.match.items():
			argument_token = extracted_argument.argument_token
			linked_argument = extracted_argument.linked_arg
			if not self.arguments[extracted_argument.get_real_complement_type()].check_plurality(argument_token, complement_types, linked_argument):
				return False

		return True



	def _check_arguments_compatibility(self, args_per_candidate: dict, argument_types: list, argument_candidates: list, referenced_token: Token):
		"""
		Checks the compatibility of the given argument types with each argument candidate
		:param args_per_candidate: the possible extracted arguments for each candidate
		:param argument_types: a list of argument types
		:param argument_candidates: the candidates for the arguments of this subcat (as list of tokens)
		:param referenced_token: the predicate of the arguments that we are after
		:return: None
		"""

		for complement_type in argument_types:
			argument = self.arguments[complement_type]

			for candidate_token in argument_candidates:
				matched_argument = argument.check_match(candidate_token, get_linked_arg(self.is_verb), referenced_token)

				if matched_argument is not None:
					args_per_candidate[candidate_token].append(matched_argument)

	@staticmethod
	def _choose_informative_positions(arguments: list):
		informative_argument_types = []

		for argument in arguments:
			found_more_informative = False

			for other_argument in arguments:
				if argument.is_more_informative(other_argument):
					found_more_informative = True
					break

			if not found_more_informative:
				informative_argument_types.append(argument)

		return informative_argument_types

	def _match_linked_arguments(self, args_that_linked_to_args: list, extraction: Extraction):
		"""
		Matches the given linked arguments based on the given match
		:param args_that_linked_to_args: a list of argument types that can be "linked" to other arguments
		:param extraction: an extraction object that includes the matches between the arguments types and the actual arguments
		:return: None
		"""

		for complement_type in difference_list(args_that_linked_to_args, extraction.get_complements()):
			arg_that_linked_to_args = self.arguments[complement_type]

			for linked_arg in difference_list(arg_that_linked_to_args.get_possible_linked_args(), [LINKED_NOM, LINKED_VERB]):
				# The linked argument (which is the referenced argument for the current complement) must appear in the given match
				# An argument that is linked for another argument can't be added without that argument
				if linked_arg not in extraction.get_complements():
					continue

				candidates = get_argument_candidates(extraction.get_argument(linked_arg).argument_token, is_verb=self.is_verb)

				for candidate_token in candidates:
					matched_argument = arg_that_linked_to_args.check_match(candidate_token, linked_arg, extraction.get_argument(linked_arg).argument_token)

					if matched_argument is not None:
						extraction.add_argument(matched_argument)

	def _get_extractions(self, args_per_candidate: dict, args_that_linked_to_args: list, referenced_token: Token, suitable_verb: str, arguments_predictor=None):
		"""
		Genetrates all the possible extractions of arguments and candidates, based on the possible arguments per candidate
		:param args_per_candidate: the possible argument types for each candidate
		:param args_that_linked_to_args: a list of argument types that can be "linked" to other arguments
		:param referenced_token: the predicate of the arguments that we are after
		:param suitable_verb: the appropriate verb for the given reference token
		:param arguments_predictor: the model-based extractor object to determine the argument type of a span (optional)
		:return: all the possible extractions for this subcat
		"""

		# Determine the arguments type of candidates with uncertainty about their complement type
		if arguments_predictor is not None and self.subcat_type == DEFAULT_SUBCAT:
			args_per_candidate = arguments_predictor.determine_args_type(args_per_candidate, referenced_token, suitable_verb, default_subcat=True)

		# Add a "None" argument option for each candidate, cause any candidate may not be an argument
		for candidate_token in args_per_candidate.keys():
			# args_per_candidate[candidate_token] = self._choose_informative_positions(args_per_candidate[candidate_token])
			args_per_candidate[candidate_token].append(None)

		candidates = args_per_candidate.keys()
		matches = [dict(zip(candidates, arguments)) for arguments in product(*args_per_candidate.values())]
		relevant_extractions = []

		for match in matches:
			extraction = Extraction(self, list(match.values()))

			# Add the linked arguments into the extraction
			self._match_linked_arguments(args_that_linked_to_args, extraction)

			# Check constraints on the current extraction
			if self._check_constraints(extraction, referenced_token):
				relevant_extractions.append(extraction)

		return relevant_extractions

	def match_arguments(self, argument_candidates: list, referenced_token: Token, suitable_verb: str, arguments_predictor=None):
		"""
		Matches the given argument candidates to the possible arguments of this subcat
		:param argument_candidates: the candidates for the arguments of this subcat (as list of tokens)
		:param referenced_token: the predicate of the arguments that we are after
		:param suitable_verb: the appropriate verb for the given reference token
		:param arguments_predictor: the model-based extractor object to determine the argument type of a span (optional)
		:return: A list of all the possible argument extractions for this subcat ([Extraction])
		"""

		# The possible arguments for each candidate
		args_per_candidate = defaultdict(list)

		# Aggregate the arguments that can be linked to other arguments (rather than NOM or VERB)
		# Those arguments can be connected in the dependency tree, to another argument of the referenced word
		args_that_linked_to_args = [arg_type for arg_type, arg in self.arguments.items() if not arg.is_linked()]

		# Check the compatability of the candidates with the required arguments first
		self._check_arguments_compatibility(args_per_candidate, self.requires, argument_candidates, referenced_token)
		if len(args_per_candidate.keys()) < len(difference_list(self.requires, args_that_linked_to_args)):
			return []

		# Then, check for the optional arguments
		self._check_arguments_compatibility(args_per_candidate, self.optionals, argument_candidates, referenced_token)

		# From possible arguments for each candidate, to possible extractions
		extractions = self._get_extractions(args_per_candidate, args_that_linked_to_args, referenced_token, suitable_verb, arguments_predictor)

		return extractions