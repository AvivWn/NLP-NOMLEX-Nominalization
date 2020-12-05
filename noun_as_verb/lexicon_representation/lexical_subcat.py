from collections import defaultdict

from spacy.tokens import Token

from .lexical_argument import LexicalArgument
from noun_as_verb.arguments_extractor.extraction.extraction import Extraction
from noun_as_verb.constants.lexicon_constants import *
from noun_as_verb.utils import difference_list


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

	def get_args_types(self):
		return self.arguments.keys()

	def get_subcat_type(self):
		return self.subcat_type

	def get_arg(self, arg_type):
		return self.arguments[arg_type]

	def is_required(self, arg_type):
		return arg_type in self.requires

	def is_default_subcat(self):
		return self.subcat_type == DEFAULT_SUBCAT

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
				if self.arguments[other_complement_type].is_det_poss_only():
					return False

			elif matched_position == POS_N_N_MOD:
				if self.arguments[other_complement_type].is_n_n_mod_only():
					return False

		return True

	def check_constraints(self, extraction: Extraction, referenced_token: Token):
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

		# The sbuject of a *verb* may follow "by" only if the verb is written in passive voice
		if self.is_verb and COMP_SUBJ in complement_types:
			subj_pos = extraction.get_argument(COMP_SUBJ).get_position()
			obj_pos = extraction.get_argument(COMP_OBJ).get_position() if COMP_OBJ in complement_types else None

			if subj_pos == "by" and obj_pos not in [POS_NSUBJPASS, None]:
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
		for arg in extraction.get_arguments():
			arg_token = arg.get_token()
			arg_type = arg.get_real_type()
			if not self.arguments[arg_type].check_plurality(arg_token, complement_types):
				return False

		return True
