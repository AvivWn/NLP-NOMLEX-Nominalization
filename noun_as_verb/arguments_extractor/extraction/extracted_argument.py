import re

from spacy.tokens import Token

from noun_as_verb.constants.lexicon_constants import *
from noun_as_verb.constants.ud_constants import *
from noun_as_verb.utils import list_to_regex
from noun_as_verb import config

class ExtractedArgument:
	argument_token: Token
	argument_span: Token

	complement_type: str
	argument_name: str		# Could be different from the real complement type

	matched_position: str
	entropy: float

	def __init__(self, argument_token: Token, complement_type, matched_position=None, entropy=None):
		self.argument_token = argument_token
		self.complement_type = complement_type
		self.argument_name = complement_type

		self.matched_position = matched_position
		self.entropy = entropy

		if matched_position:
			self.argument_span = self.as_span()
			self.trimmed_argument_sapn = self.as_span(trim_all=True)

	def get_real_type(self):
		return self.complement_type

	def get_head_idx(self):
		return self.argument_token.i

	def get_name(self):
		if self.argument_name in [NOM_TYPE_NONE, NOM_TYPE_VERB_NOM]:
			return None

		return self.argument_name

	def get_token(self):
		return self.argument_token

	def get_position(self):
		return self.matched_position

	def get_properties(self):
		return self.argument_token, self.complement_type, self.matched_position, self.argument_name

	def get_entropy(self):
		return self.entropy

	def get_span(self):
		return self.trimmed_argument_sapn

	def set_matched_position(self, matched_position):
		self.matched_position = matched_position
		self.argument_span = self.as_span()
		self.trimmed_argument_sapn = self.as_span(trim_all=True)

	def set_name(self, argument_name):
		self.argument_name = argument_name
		self.argument_span = self.as_span()
		self.trimmed_argument_sapn = self.as_span(trim_all=True)



	def is_more_informative(self, other_argument):
		# Wether the given argument is more iformative than this argument

		complement_type = self.get_real_type()
		argument_name = self.argument_name
		matched_position = self.matched_position

		other_complement_type = other_argument.get_real_type()
		other_argument_name = other_argument.argument_name
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

		# FOR-TO-INF is preferable than a standard TO-INF phrase
		if complement_type.startswith(COMP_TO_INF) and other_complement_type.startswith("FOR-"):
			return True

		# An arbitrary controlled argument should be avoided when the standard arguemnt is an option
		if "-ARBC" in argument_name and argument_name.replace("-ARBC", "") == other_argument_name:
			return True

		# A possessive controlled argument is preferable over standard one (POSSING vs ING)
		if "POSS" in other_argument_name and other_argument_name.replace("POSS", "") == argument_name:
			return True

		# FOR-TO-INF is preferable over standard preposition phrase
		# We want that FOR-TO-INF argument won't be tagged as FOR-NP (like in PP)
		if matched_position.islower() and other_complement_type == COMP_FOR_TO_INF:
			return True

		# A complex prepositional phrase is preferable than a standard gerund phrase or infinitival phrase
		# Even when the complex PP contain a gerund, it is refered not as a gerund phrase
		# The PP might be P-HOW-S with a gerund, or WH-S and so on
		# Currently P-ING is also preferable over just ING
		if matched_position in [POS_ING, POS_TO_INF] and other_matched_position.islower():
			return True

		# Prefix of preposition is preferable over ADVP, for 2-worded prepositions like "next to"
		if complement_type == COMP_ADVP and other_matched_position.islower():
			return True

		# Sometimes an argument with "P-" can be the same as the argument without it
		if "P-" + complement_type == other_complement_type:
			return True

		# AS-ING is probably more specific than AS-ADJ (like in as being ill)
		if complement_type == COMP_AS_ADJP and "AS-ING" in other_complement_type:
			return True

		return False



	def as_span(self, trim_argument=True, trim_all=False):
		arg_indices = [self.argument_token.i]

		# NOM complement stays as NOM
		if self.matched_position != POS_NOM:
			arg_indices = self.argument_token._.subtree_indices

		assert arg_indices != [], Exception("Found an empty argument!")

		dependency_tree = self.argument_token.doc
		start_span_index = min(arg_indices)
		end_span_index = max(arg_indices) + 1
		arg_span = dependency_tree[start_span_index: end_span_index]

		# Remove possessive suffixes and punctuation endings
		ends_with_poss = arg_span[-1].orth_.lower() in ["'s", "s'", "’s", "s’"] and self.matched_position in [POS_DET_POSS, POS_NSUBJ, POS_N_N_MOD]
		ends_with_punct = arg_span[-1].pos_ == UPOS_PUNCT
		if ends_with_poss or ends_with_punct:
			arg_span = arg_span[:-1]

		# Remove prepositional prefixes
		if trim_all or (trim_argument and self.matched_position.islower() and re.match(f'^PP|^P-|^FOR-|^AS-|{COMP_PART}', self.argument_name) is None):
			only_preposition = re.sub(list_to_regex(WHERE_WHEN_OPTIONS + WH_VERB_OPTIONS + HOW_TO_OPTIONS + HOW_OPTIONS, "|"), '', self.matched_position).strip()

			if only_preposition != "" and arg_span.text.startswith(only_preposition):
				arg_span = arg_span[len(only_preposition.split(" ")):]

		# Ignore empty arguments after the trimming
		if len(arg_span) == 0:
			return None

		return arg_span