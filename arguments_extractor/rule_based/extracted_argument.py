import re
from copy import deepcopy

from spacy.tokens import Token

from arguments_extractor.constants.lexicon_constants import *
from arguments_extractor.constants.ud_constants import *
from arguments_extractor.utils import list_to_regex
from arguments_extractor import config

class ExtractedArgument:
	argument_token: Token
	matched_position: str
	linked_arg: str
	argument_name: str		# Could be different from the real complement type
	real_complement_type: str

	def __init__(self, argument_token: Token, lexical_argument, linked_arg: str, matched_position=None, complement_type=None):
		self.argument_token = argument_token
		self.lexical_argument = lexical_argument
		self.matched_position = matched_position
		self.linked_arg = linked_arg

		if complement_type is None:
			self.argument_name = lexical_argument.get_complement_type()
		else:
			self.argument_name = complement_type

	def get_real_complement_type(self):
		return self.lexical_argument.get_complement_type()

	def get_argument_idx(self):
		return self.argument_token.i

	def get_properties(self):
		return self.argument_token, self.lexical_argument, self.matched_position, self.linked_arg, self.argument_name

	def set_matched_position(self, matched_position):
		self.matched_position = matched_position

	def set_argument_name(self, argument_name):
		self.argument_name = argument_name

	def is_more_informative(self, other_argument):
		# Wether the given argument is more iformative than this argument

		complement_type = self.get_real_complement_type()
		matched_position = self.matched_position

		other_complement_type = other_argument.get_real_complement_type()
		other_matched_position = other_argument.matched_position

		if matched_position.islower() and other_matched_position.islower():
			# We should take the prefix argument with the most informative prefix (meaning "about what" and not just "about")
			if len(matched_position) < len(other_matched_position):
				return True

			# Simple preposition argument should not appear if there is another type of complement to the same argument
			if complement_type in [COMP_PP, COMP_PP1, COMP_PP2] and other_complement_type.startswith("P-"):
				return True

			# AS argument is more preferable over other prepositional phrases
			if not complement_type.startswith("AS-") and other_complement_type.startswith("AS-"):
				return True

		# FOR-TO-INF is preferable than a standard TO-INF phrase (shouldn't occur due to the lexicon structure)
		if complement_type.startswith(COMP_TO_INF) and other_complement_type.startswith("FOR-"):
			return True

		# A complex prepositional phrase is preferable than a standard gerund phrase
		# Even when the complex PP contain a gerund, it is refered not as a gerund phrase
		# The PP might be P-HOW-S with a gerund, or WH-S and so on
		if matched_position == POS_ING and other_matched_position.islower():
			return True

		# Prefix of preposition is preferable over ADVP
		if complement_type == COMP_ADVP and other_matched_position.islower():
			return True

		return False

	def get_argument_span(self, extraction_token_indices):
		relevant_children = [self.argument_token.i]

		# NOM complement stays as NOM
		if self.matched_position != POS_NOM:
			# Find all the child-tokens that aren't the root of another complement
			for child_token in self.argument_token.children:
				if child_token.i not in extraction_token_indices:
					relevant_children += [sub_token.i for sub_token in child_token.subtree]

		# Each argument must be a span without holes
		relevant_children.sort()
		tmp_children = deepcopy(relevant_children)
		for i in range(1, len(tmp_children)):
			if tmp_children[i] != tmp_children[i - 1] + 1:
				relevant_children = relevant_children[1:]

		if config.DEBUG and relevant_children == []:
			raise Exception("Found an empty argument!")

		dependency_tree = self.argument_token.doc
		start_span_index = min(relevant_children)
		end_span_index = max(relevant_children) + 1
		argument_span = dependency_tree[start_span_index: end_span_index]

		# Remove possessive suffixes
		if argument_span[-1].pos_ == UPOS_PUNCT or (argument_span[-1].orth_.lower() in ["'s", "s'", "’s", "s’"] and self.matched_position in [POS_DET_POSS, POS_NSUBJ, POS_N_N_MOD]):
			argument_span = argument_span[:-1]

		# Remove prepositionsal prefixes
		if config.CLEAN_NP and self.matched_position.islower() and re.match(f'^PP|^P-|^FOR-|^AS-|{COMP_PART}', self.argument_name) is None:
			only_preposition = re.sub(list_to_regex(WHERE_WHEN_OPTIONS + WH_VERB_OPTIONS + HOW_TO_OPTIONS + HOW_OPTIONS, "|"), '', self.matched_position).strip()

			if only_preposition != "":
				argument_span = argument_span[len(only_preposition.split(" ")):]

		return argument_span