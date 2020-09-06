import re
from collections import defaultdict

from spacy.tokens import Token

from arguments_extractor.rule_based.extracted_argument import ExtractedArgument
from arguments_extractor.rule_based.utils import check_relations, relation_to_position, get_word_in_relation
from arguments_extractor.constants.lexicon_constants import *
from arguments_extractor.constants.ud_constants import *
from arguments_extractor.utils import list_to_regex, get_linked_arg

class LexicalArgument:
	# The type of the argument\complement
	complement_type: str

	# Possible positions
	positions: dict

	# Other constraints on the argument
	prefix_pattern: dict
	illegal_prefix_pattern: dict
	root_pattern: dict
	root_upostags: dict
	root_urelations: dict
	constraints: dict

	is_verb: bool

	def __init__(self, argument_info, complement_type, is_verb):
		self.complement_type = complement_type
		self.positions = defaultdict(list)

		self.prefix_pattern = defaultdict(str)
		self.illegal_prefix_pattern = defaultdict(str)
		self.root_pattern = defaultdict(str)
		self.root_upostags = defaultdict(list)
		self.root_urelations = defaultdict(list)
		self.constraints = defaultdict(list)
		self.including = defaultdict(str)

		for linked_arg in argument_info.keys():
			self.positions[linked_arg] = argument_info[linked_arg].get(ARG_POSITIONS, [])
			self.root_upostags[linked_arg] = argument_info[linked_arg].get(ARG_ROOT_UPOSTAGS, [])
			self.root_urelations[linked_arg] = argument_info[linked_arg].get(ARG_ROOT_URELATIONS, [])
			self.constraints[linked_arg] = argument_info[linked_arg].get(ARG_CONSTRAINTS, [])
			self.including[linked_arg] = argument_info[linked_arg].get(ARG_INCLUDING, None)

			# Translate the patterns list into regex patterns
			self.prefix_pattern[linked_arg] = list_to_regex(argument_info[linked_arg].get(ARG_PREFIXES, []), "|", start_constraint="^", end_constraint=" ")
			self.illegal_prefix_pattern[linked_arg] = list_to_regex(argument_info[linked_arg].get(ARG_ILLEGAL_PREFIXES, []), "|", start_constraint="^", end_constraint=" ")
			self.root_pattern[linked_arg] = list_to_regex(argument_info[linked_arg].get(ARG_ROOT_PATTERNS, []), "|")

		self.is_verb = is_verb
		self.linked_args = list(argument_info.keys())

	def get_complement_type(self):
		return self.complement_type

	def get_possible_linked_args(self):
		return self.linked_args

	def is_linked(self):
		return self.get_possible_linked_args() == [get_linked_arg(self.is_verb)]



	# Argument constraints

	def is_det_poss_only(self, linked_arg: str):
		return ARG_CONSTRAINT_DET_POSS_NO_OTHER_OBJ in self.constraints[linked_arg]

	def is_n_n_mod_only(self, linked_arg: str):
		return ARG_CONSTRAINT_N_N_MOD_NO_OTHER_OBJ in self.constraints[linked_arg]

	def check_plurality(self, candidate_token: Token, other_complements: list, linked_arg: str):
		if ARG_CONSTRAINT_PLURAL not in self.constraints[linked_arg]:
			return True

		if COMP_SECOND_SUBJ in other_complements:
			return True

		if candidate_token.tag_ in [TAG_NNS, TAG_NNPS]:
			return True

		return False


	def _check_root(self, candidate_token: Token, matched_argument: ExtractedArgument, linked_arg: str):
		"""
		Checks that the constraints on the root according to this argument works for the given root word
		:param candidate_token: a token candidate for this argument
		:param matched_argument: The appropriate argument object for this lexical argument
		:param linked_arg: the linked argument (usually the nominalization [NON] or the verb [VERB])
		:return: True if the root doesn't contradict the root constraints of this argument, and False otherwise
		"""

		# Check whether the matched position is a multi-word preposition and the candidate token is part of the preposition prefix
		# If so, then the "root" of the candidate should be the nearest connected token *after the preposition*, for the purpose of the next tests
		# Example- "... with regard to the man". The candidate token will be "regard". But we must check the constraints over "man"
		matched_position = matched_argument.matched_position
		if matched_position.islower():
			candidate_index_in_arg = candidate_token.i - candidate_token._.subtree_indices[0]
			prep_length = len(matched_position.split(" "))

			if prep_length > 1 and candidate_index_in_arg < prep_length:
				#@TODO- is it right to use "wild card" relation here?
				end_of_preposition_idx = candidate_token._.subtree_indices[0] + prep_length
				candidate_token = get_word_in_relation(candidate_token, URELATION_ANY, start_index=end_of_preposition_idx)

				if candidate_token is None:
					return False

		if not check_relations(candidate_token, self.root_urelations[linked_arg]):
			return False

		# ING and TO-INF complements may include be-form verb instead of the main verb of those complements
		# In such cases the "be" verb isn't the root, and the real root doesn't obey its contraints (cause it isn't the verb)
		if COMP_TO_INF in self.complement_type or "ING" in self.complement_type:
			needed_be_form = "be" if COMP_TO_INF in self.complement_type else "being"
			if check_relations(candidate_token, [URELATION_COP + "_" + needed_be_form]):
				return True

		if self.root_upostags[linked_arg] != [] and candidate_token.pos_ not in self.root_upostags[linked_arg]:
			return False

		#@TODO- can a determiner that isn't possessive pronoun be an NP argument?
		if candidate_token.pos_ == UPOS_DET and self.complement_type in [COMP_SUBJ, COMP_OBJ, COMP_IND_OBJ, COMP_NP] and candidate_token.orth_.lower() not in POSSESIVE_OPTIONS:
			return False

		if self.root_pattern[linked_arg] != "" and not re.search(self.root_pattern[linked_arg], candidate_token.orth_.lower(), re.M):
			return False

		return True

	@staticmethod
	def _is_possessive(token: Token):
		token_subtree = token._.subtree_text

		if token_subtree.lower().endswith("'s") or token_subtree.lower().endswith("' s"):
			return True

		if token_subtree.lower().endswith("s'") or token_subtree.lower().endswith("s '"):
			return True

		if token.orth_.lower() in POSSESIVE_OPTIONS:
			return True

		return False

	def _check_constraints(self, candidate_token: Token, matched_argument: ExtractedArgument, linked_arg: str):
		"""
		Checks whether the given candidate is compatible with the constraints of this argument
		:param candidate_token: a token candidate for this argument
		:param matched_argument: The appropriate argument object for this lexical argument
		:param linked_arg: the linked argument (usually the nominalization [NON] or the verb [VERB])
		:return: True if the candidate doesn't contradict the constraints, and False otherwise
		"""

		# Checks the constraints on the root
		if not self._check_root(candidate_token, matched_argument, linked_arg):
			return False

		####################################
		# Check the boolean constraints

		# Handle optional possessive sub-argument
		if ARG_CONSTRAINT_OPTIONAL_POSSESSIVE in self.constraints[linked_arg]:
			founded_possessive = False

			for relation in [URELATION_NMOD_POSS, URELATION_NSUBJ]:
				founded_token = get_word_in_relation(candidate_token, relation)

				if founded_token is None:
					continue

				if self._is_possessive(founded_token):
					founded_possessive = True
					break

			# Change the argument name of the given matched argument object, if needed

			# Possessive should be included
			if founded_possessive and "POSSING" not in self.complement_type:
				matched_argument.set_name(self.complement_type.replace("ING", "POSSING"))

			# Possessive should be excluded
			elif not founded_possessive and "POSSING" in self.complement_type:
				matched_argument.set_name(self.complement_type.replace("POSSING", "ING-ARBC"))

		return True



	def _check_position(self, position: str, candidate_token: Token, matched_argument: ExtractedArgument, linked_arg: str):
		"""
		Checks whether the given candidate word and this argument are compatible and can occur as the given position
		:param position: a possible position for the candidate (from constant position names, like PREFIX, DOBJ and more)
		:param candidate_token: a token candidate for this argument
		:param matched_argument: The appropriate argument object for this lexical argument
		:param linked_arg: the linked argument (usually the nominalization [NON] or the verb [VERB])
		:return: True if the three (this argument and the given position and candidate) are compatible with the regard to the linked argument, and False otherwise
		"""

		matched_position = position

		# Check that this argument is compatible with that position
		if position not in self.positions[linked_arg]:
			return False

		if position == POS_PREFIX:
			# Empty pattern means that this argument isn't compatible with any prefix position
			if self.prefix_pattern[linked_arg] == "":
				return False

			# Check whether the candidate is compatible with the prefix pattern
			matched_position = re.search(self.prefix_pattern[linked_arg], candidate_token._.subtree_text + " ", re.M)
			if matched_position is None:
				return False

			matched_position = matched_position.group().strip()

			# The prefix cannot be the entire argument
			if len(matched_position.split()) == len(candidate_token._.subtree_text.split()):
				return False

		# a complement without a standard prefix position may also required a specific prefix constraint
		elif ARG_CONSTRAINT_REQUIRED_PREFIX in self.constraints[linked_arg] and self.prefix_pattern[linked_arg] != "" \
					and re.search(self.prefix_pattern[linked_arg], candidate_token._.subtree_text + " ", re.M) is None:
				return False

		# Check wether the candidate isn't compatible with the *illegal* prefix pattern
		if self.illegal_prefix_pattern[linked_arg] != "" and \
				re.search(self.illegal_prefix_pattern[linked_arg], candidate_token._.subtree_text + " ", re.M) is not None:
			return False

		# Update the matched position of the given matched argument
		matched_argument.set_matched_position(matched_position)

		# Check the compatibility between the candidate and this argument
		return self._check_constraints(candidate_token, matched_argument, linked_arg)

	def check_match(self, candidate_token: Token, linked_arg: str, referenced_token: Token):
		"""
		Checks whether the given candidate matches to to this argument
		:param candidate_token: a token candidate for this argument
		:param linked_arg: the linked argument (usually the nominalization [NON] or the verb [VERB])
		:param referenced_token: the predicate of the arguments that we are after
		:return: the matched argument (ExtractedArgument object) if a match was found, or None otherwise
		"""

		if linked_arg not in self.get_possible_linked_args():
			return None

		matched_argument = ExtractedArgument(candidate_token, self, linked_arg)

		# Get the possible "position" type for the candidate (like DET-POSS, PREFIX and so on)
		# Based on the dependency relation that connects the candidate to the rest of the tree (its head relation)
		possible_positions = relation_to_position(candidate_token, referenced_token, self.is_verb)

		# Check the compatibility of each position with this argument and the candidate
		for position in possible_positions:
			if self._check_position(position, candidate_token, matched_argument, linked_arg):
				return matched_argument

		return None