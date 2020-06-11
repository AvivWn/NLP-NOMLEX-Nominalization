from .argument import *

class Subcat:
	arguments = defaultdict(Argument)
	requires: list
	optionals: list
	constraints: list
	not_constraints: dict
	is_verb: bool

	def __init__(self, subcat_info: dict, is_verb):
		self.arguments = defaultdict(Argument)
		for complement_type in difference_list(subcat_info.keys(), [SUBCAT_REQUIRED, SUBCAT_OPTIONAL, SUBCAT_NOT, SUBCAT_CONSTRAINTS]):
			self.arguments[complement_type] = Argument(subcat_info[complement_type], is_verb)

		self.requires = subcat_info.get(SUBCAT_REQUIRED, [])
		self.optionals = difference_list(self.arguments.keys(), subcat_info.get(SUBCAT_REQUIRED, []))
		self.not_constraints = subcat_info.get(SUBCAT_NOT, [])
		self.constraints = subcat_info.get(SUBCAT_CONSTRAINTS, [])
		self.is_verb = is_verb

	def check_uniqueness(self, arguments: list, positions: list):
		"""
		Checks that different arguments of noms don't appear using the same positions (relevant to specific positions)
		And that any argument don't appear more than once
		:param arguments: A list of argument types
		:param positions: A list of position types
		:return: True if the given matching is compatible with the "Uniqueness Principle", and False otherwise
		"""

		# Any argument can appear once
		if len(arguments) != len(set(arguments)):
			return False

		if self.is_verb:
			return True

		# The next positions can appear at least once
		for position in [POS_POSS, "of"]:
			if list(positions).count(position) > 1:
				return False

		return True

	def check_ordering(self, matching: dict, referenced_word_index: int):
		"""
		Checks that the order of the pre-nom arguments is as "SUBJECT > INDIRECT OBJECT > DIRECT OBJECT > OBLIQUE" ("Ordering Constraint", NOMLEX manual)
		This constraint is relevant only to nominalizations
		:param matching: The matches between the arguments types and the actual arguments ({type: index})
		:param referenced_word_index: the index of the word that we are looking for its arguments, like nom or verb
		:return: True if the given matching is compatible with the "Ordering Constraint", and False otherwise
		"""

		if self.is_verb:
			return True

		subject_index = matching.get(COMP_SUBJ, referenced_word_index)
		ind_object_index = matching.get(COMP_IND_OBJ, referenced_word_index)
		object_index = matching.get(COMP_OBJ, referenced_word_index)
		other_indexes = [referenced_word_index]

		for complement_type, root_index in matching.items():
			if complement_type not in [COMP_SUBJ, COMP_OBJ, COMP_IND_OBJ]:
				other_indexes.append(root_index)

		# Subject must appear before ind-object and object and others, if it appears before nom
		if subject_index < referenced_word_index and subject_index > min([ind_object_index, object_index] + other_indexes):
			return False

		# Ind-object must appear before object and any others, if it appears before nom
		if ind_object_index < referenced_word_index and ind_object_index > min([object_index] + other_indexes):
			return False

		# Object must appear before any others, if it appears before nom
		if ind_object_index < referenced_word_index and ind_object_index > min(other_indexes):
			return False

		return True

	def is_in_not(self, arg_and_pos: list):
		"""
		Checks whether the given match between arguments and positions appear in the NOT constraints
		:param arg_and_pos: list of pairs of arguments and positions ([(arg, pos)])
		:return: True if this matching don't appear in the NOT constraints of this subcat, and False otherwise
		"""

		if self.not_constraints == []:
			return False

		for not_constraint in self.not_constraints:
			for complement_type, position in arg_and_pos:
				# A NOT constraint may relate only for few complement types
				if complement_type not in not_constraint.keys():
					continue

				# Is this position allowed for this complement?
				if position not in not_constraint[complement_type]:
					return False

		return True

	def check_if_no_other_object(self, argument_types: list, positions: list):
		"""
		Checks that the compatibility of the single object argument of nominalizations (not relevant to verbs)
		This function has any meaning only when there is signle object argument
		:param argument_types: a list of founded argument types
		:param positions: a list of the founded position of each argument
		:return: True if the single argument is also
		"""

		if self.is_verb:
			return True

		object_args = []

		# Get all the "objects" that are arguments of the nominalization
		for object_candidate in [COMP_OBJ, COMP_IND_OBJ, COMP_SUBJ]:
			if object_candidate in argument_types:
				object_args.append(object_candidate)

		if len(object_args) != 1:
			return True

		# Only one argument was founded
		argument_index = argument_types.index(object_args[0])
		argument_type = argument_types[argument_index]
		position = positions[argument_index]

		# Check if there isn't any other argument that should get that position when it is the only object
		for other_argument_type in difference_list(self.arguments.keys(), [argument_type]):
			if position == POS_POSS:
				if self.arguments[other_argument_type].is_det_poss_only(get_linked_arg(self.is_verb)):
					return False

			elif position == POS_COMPOUND:
				if self.arguments[other_argument_type].is_n_n_mod_only(get_linked_arg(self.is_verb)):
					return False

		return True

	def check_constraints(self, matching: dict, referenced_word_index: int):
		"""
		Checks whether the given subcat matching is compatible with the constraints of this subcat
		:param matching: The matches between the arguments types and the actual arguments ({type: index})
		:param referenced_word_index: the index of the word that we are looking for its arguments, like nom or verb
		:return: True if the candidate doesn't contradict the constraints, and False otherwise
		"""

		# Check that all the required arguments appear in the given matching
		if not set(self.requires).issubset(matching.keys()):
			return False

		# Check that the given matching satisfied the "Ordering Constraint" (NOMLEX manual)
		if not self.check_ordering(matching, referenced_word_index):
			return False

		####################################
		# Check the boolean constraints

		if SUBCAT_CONSTRAINT_ADVP_OR_ADJP in self.constraints:
			if COMP_ADVP not in matching and COMP_ADJP not in matching:
				return False

		return True

	def check_arguments_compatibility(self, args_per_candidate: dict, dependency_tree: list, argument_types: list, argument_candidates: list):
		"""
		Checks the compatibility of the given argument types with each argument candidate
		:param args_per_candidate: The possible argument types for each candidate
		:param dependency_tree: the appropriate dependency tree for a sentence
		:param argument_types: a list of argument types
		:param argument_candidates: the candidates for the arguments of this subcat (as list of root indexes)
		:return: None
		"""

		for complement_type in argument_types:
			argument = self.arguments[complement_type]

			for candidate_index in argument_candidates:
				is_matched, matched_position = argument.check_matching(dependency_tree, candidate_index, get_linked_arg(self.is_verb))

				if is_matched:
					args_per_candidate[candidate_index].append((complement_type, matched_position))

	def match_linked_arguments(self, dependency_tree: list, args_that_linked_to_args: list, matching: dict):
		"""
		Matches the given linked arguments based on the given matching
		:param dependency_tree: the appropriate dependency tree for a sentence
		:param args_that_linked_to_args: A list of argument types that can be "linked" to other arguments
		:param matching: The matches between the arguments types and the actual arguments ({type: index})
		:return: None
		"""

		for complement_type in difference_list(args_that_linked_to_args, matching.keys()):
			arg_that_linked_to_args = self.arguments[complement_type]

			for linked_arg in difference_list(arg_that_linked_to_args.get_possible_linked_args(), [LINKED_NOM, LINKED_VERB]):
				candidates = get_argument_candidates(dependency_tree, matching[linked_arg])

				for candidate_index in candidates:
					is_matched, _ = arg_that_linked_to_args.check_matching(dependency_tree, candidate_index,get_linked_arg(self.is_verb))

					if is_matched:
						matching[complement_type] = candidate_index

	def get_matchings(self, args_per_candidate: dict, dependency_tree: list, args_that_linked_to_args: list, referenced_word_index: int):
		"""
		Genetrates all the possible matching of arguments and candidates, based on the possible arguments per candidate
		:param args_per_candidate: The possible argument types for each candidate
		:param dependency_tree: the appropriate dependency tree for a sentence
		:param args_that_linked_to_args: A list of argument types that can be "linked" to other arguments
		:param referenced_word_index: the index of the word that we are looking for its arguments, like nom or verb
		:return: All the possible matching for this subcat
		"""

		# Retrieve all the possible matchings of candidate arguments with this subcat's arguments and sort it by the number of arguments
		matchings = []
		for arg_and_pos in product(*args_per_candidate.values()):
			# Check that the arguments and positions not appear in the NOT position constraints
			if self.is_in_not(list(arg_and_pos)):
				continue

			arguments = [arg for arg, pos in arg_and_pos]
			positions = [pos for arg, pos in arg_and_pos]

			# Check that the arguments and positions satisfied the "Uniqueness Principle" (NOMLEX manual)
			if not self.check_uniqueness(arguments, positions):
				continue

			# Check that the position of a single object (only if it single) is legitimate
			if not self.check_if_no_other_object(arguments, positions):
				continue

			candidates = args_per_candidate.keys()
			matchings.append(dict(zip(candidates, arguments)))

		matchings = sorted(matchings, key=lambda k: len(k.keys()), reverse=True)

		# Get the correct matchings, meaning the ones that fulfills the constraints of this subcat
		correct_matchings = []
		for matching in matchings:
			matching = reverse_dict(matching)
			self.match_linked_arguments(dependency_tree, args_that_linked_to_args, matching)

			# Did we find an extraction that includes that matching?
			is_sub_matching = any([matching.items() <= other_matching.items() for other_matching in correct_matchings])

			# Check constraints on that extraction
			if not is_sub_matching and self.check_constraints(matching, referenced_word_index):
				correct_matchings.append(matching)

		return correct_matchings

	def match_arguments(self, dependency_tree: list, argument_candidates: list, referenced_word_index: int):
		"""
		Matches the given argument candidates to the possible arguments of this subcat
		:param dependency_tree: the appropriate dependency tree for a sentence
		:param argument_candidates: the candidates for the arguments of this subcat (as list of root indexes)
		:param referenced_word_index: the index of the word that we are looking for its arguments, like nom or verb
		:return: a list of all the founded argument matching for this subcat ([{ARG: root_index}])
		"""

		# The possible arguments for each candidate
		args_per_candidate = defaultdict(list)

		# Aggregate the arguments that can be linked to other arguments (rather than NOM or VERB)
		# Those arguments can be connected in the dependency tree, to an argument
		args_that_linked_to_args = [arg_type for arg_type, arg in self.arguments.items() if not arg.is_linked()]

		# Check the compatability of the candidates with the required arguments first
		self.check_arguments_compatibility(args_per_candidate, dependency_tree, self.requires, argument_candidates)
		if len(args_per_candidate.keys()) < len(difference_list(self.requires, args_that_linked_to_args)):
			return []

		# Then, check for the optional arguments
		self.check_arguments_compatibility(args_per_candidate, dependency_tree, self.optionals, argument_candidates)

		matchings = self.get_matchings(args_per_candidate, dependency_tree, args_that_linked_to_args, referenced_word_index)

		return matchings