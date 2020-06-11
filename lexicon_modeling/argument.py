from .utils import *

class Argument:
	# Possible positions as two different lists
	constant_positions: dict
	prefix_pattern: dict

	# Other constraints on the argument
	illegal_prefix_pattern: dict
	root_pattern: dict
	root_upostags: dict
	root_urelations: dict
	constraints: dict

	is_verb: bool

	def __init__(self, argument_info, is_verb):
		self.constant_positions = defaultdict(list)
		self.prefix_pattern = defaultdict(str)
		self.illegal_prefix_pattern = defaultdict(str)
		self.root_pattern = defaultdict(str)
		self.root_upostags = defaultdict(list)
		self.root_urelations = defaultdict(list)
		self.constraints = defaultdict(list)

		for linked_arg in argument_info.keys():
			self.constant_positions[linked_arg] = argument_info[linked_arg].get(ARG_CONSTANTS, [])
			self.root_upostags[linked_arg] = argument_info[linked_arg].get(ARG_ROOT_UPOSTAGS, [])
			self.root_urelations[linked_arg] = argument_info[linked_arg].get(ARG_ROOT_RELATIONS, [])
			self.constraints[linked_arg] = argument_info[linked_arg].get(ARG_CONSTRAINTS, [])

			# Translate the patterns list into regex patterns
			self.prefix_pattern[linked_arg] = list_to_regex(argument_info[linked_arg].get(ARG_PREFIXES, []), "|", start_constraint="^")
			self.illegal_prefix_pattern[linked_arg] = list_to_regex(argument_info[linked_arg].get(ARG_ILLEGAL_PREFIXES, []), "|", start_constraint="^")
			self.root_pattern[linked_arg] = list_to_regex(argument_info[linked_arg].get(ARG_ROOT_PATTERNS, []), "|")

		self.is_verb = is_verb
		self.linked_args = list(argument_info.keys())

	def get_possible_linked_args(self):
		return self.linked_args

	def is_linked(self):
		return self.get_possible_linked_args() == [get_linked_arg(self.is_verb)]

	def is_det_poss_only(self, linked_arg: str):
		return ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ in self.constraints[linked_arg]

	def is_n_n_mod_only(self, linked_arg: str):
		return ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ in self.constraints[linked_arg]

	def check_root(self, dependency_tree: list, root_info: dict, linked_arg: str):
		"""
		Checks that the constraints on the root according to this argument works for the given root word
		:param dependency_tree: the appropriate dependency tree for a sentence
		:param root_info: the info of the root of an arugment in the dependency tree
		:param linked_arg: the linked argument (usually the nominalization [NON] or the verb [VERB])
		:return: True if the root doesn't contradict the root constraints of this argument, and False otherwise
		"""

		if self.root_upostags[linked_arg] != [] and root_info[WORD_UPOS_TAG] not in self.root_upostags[linked_arg]:
			return False

		if self.root_pattern[linked_arg] != "" and not re.search(self.root_pattern[linked_arg], root_info[WORD_TEXT].lower(), re.M):
			return False

		if not check_relations(dependency_tree, root_info, self.root_urelations[linked_arg]):
			return False

		return True

	def check_constraints(self, dependency_tree: list, candidate_info: dict, linked_arg: str):
		"""
		Checks whether the given candidate is compatible with the constraints of this argument
		:param dependency_tree: the appropriate dependency tree for a sentence
		:param candidate_info: the info of the candidate in the dependency tree (this is the root of the argument)
		:param linked_arg: the linked argument (usually the nominalization [NON] or the verb [VERB])
		:return: True if the candidate doesn't contradict the constraints, and False otherwise
		"""

		# Checks the constraints on the root
		if not self.check_root(dependency_tree, candidate_info, linked_arg):
			return False

		####################################
		# Check the boolean constraints

		# Check the possessive constraint
		if ARG_CONSTRAINT_POSSESSIVE in self.constraints[linked_arg]:
			if not candidate_info[WORD_SUB_TREE].lower().endswith("'s") and not candidate_info[WORD_TEXT].lower() in POSSESIVE_OPTIONS:
				return False

		return True

	def check_position(self, dependency_tree: list, position: str, candidate_info: dict, linked_arg: str):
		"""
		Checks whether the given candidate word and this argument are compatible and can occur as the given position
		:param dependency_tree: the appropriate dependency tree for a sentence
		:param position: a possible position for the candidate (prefix or constant)
		:param candidate_info: the info of the candidate in the dependency tree (this is the root of the argument)
		:param linked_arg: the linked argument (usually the nominalization [NON] or the verb [VERB])
		:return: True if the three (this argument and the given position and candidate) are compatible, and False otherwise
		:return: is_match (bool), matched_position (str)
		 		 is_match is True if there is a match between the candidate to this argument, and False otherwise
				 matched_position is the position in which the candidate and this argument matched, and None if they didn't with any such postion
		"""

		matched_position = position

		if position == POS_PREFIX:
			# Empty pattern means that this argument isn't compatible with any prefix position
			if self.prefix_pattern[linked_arg] == "":
				return False, None

			# Check whether the candidate is compatible with the prefix pattern
			matched_position = re.search(self.prefix_pattern[linked_arg], candidate_info[WORD_SUB_TREE],re.M)
			if matched_position is None:
				return False, None

			# Check wether the candidate isn't compatible with the *illegal* prefix pattern
			if self.illegal_prefix_pattern[linked_arg] != "" and re.search(self.illegal_prefix_pattern[linked_arg], candidate_info[WORD_SUB_TREE],re.M) is not None:
				return False, None

			matched_position = matched_position.group()

		# Otherwise, this is a constant position; Check that this argument is compatible with that position
		elif position not in self.constant_positions[linked_arg]:
			return False, None

		# Check the compatibility between the candidate and this argument
		return True, matched_position#self.check_constraints(dependency_tree, candidate_info, linked_arg), matched_position

	def check_matching(self, dependency_tree: list, candidate_index: int, linked_arg: str):
		"""
		Checks whether the given candidate matches to to this argument
		:param dependency_tree: the appropriate dependency tree for a sentence
		:param candidate_index: the index of the candidate argument in the dependency tree (the root index)
		:param linked_arg: the linked argument (usually the nominalization [NON] or the verb [VERB])
		:return: is_match (bool), matched_position (str)
				 is_match is True if there is a match between the candidate to this argument, and False otherwise
				 matched_position is the position in which the candidate and this argument matched, and None if they didn't with any such postion
		"""

		if linked_arg not in self.get_possible_linked_args():
			return False, None

		candidate_info = dependency_tree[candidate_index]

		# Get the possible "position" type for the candidate (like DET-POSS, of, for and so on)
		# Based on the dependency relation that connects the candidate to the rest of the tree (its head relation)
		possible_positions = relation_to_position(candidate_info, self.is_verb)

		# Check the compatibility of each position with this argument and the candidate
		for position in possible_positions:
			is_match, matched_position = self.check_position(dependency_tree, position, candidate_info, linked_arg)

			if is_match:
				return True, matched_position

		return False, None