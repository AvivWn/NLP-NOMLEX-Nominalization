from collections import defaultdict
from spacy.tokens import Token

from arguments_extractor.rule_based.subcat import Subcat
from arguments_extractor.constants.lexicon_constants import *

class Entry:
	orth: str
	subcats = defaultdict(Subcat)
	next = None
	plural = None
	nom = None
	singular = None
	plural_freq = None
	is_verb: bool

	def __init__(self, entry: dict, is_verb):
		self.orth = entry[ENT_ORTH]
		self.nom_type = entry.get(ENT_NOM_TYPE, {})

		self.subcats = defaultdict(Subcat)
		for subcat_type in entry.get(ENT_VERB_SUBC, {}).keys():
			self.subcats[subcat_type] = Subcat(entry[ENT_VERB_SUBC][subcat_type], is_verb)

		self.next = entry.get(ENT_NEXT, None)
		self.plural = entry.get(ENT_PLURAL, None)
		self.nom = entry.get(ENT_NOM, None)
		self.singular = entry.get(ENT_SINGULAR, None)
		self.plural_freq = entry.get(ENT_PLURAL_FREQ, None)
		self.is_verb = is_verb

	def set_next(self, lexicon):
		self.next = lexicon.get_entry(self.next)

	def match_arguments(self, argument_candidates: list, referenced_token: Token):
		"""
		Matches the given argument candidates to the possible arguments of all the entries with the same orth (this, next and so on)
		:param argument_candidates: the candidates for the arguments of this entry (as list of tokens)
		:param referenced_token: the predicate of the arguments that we are after
		:return: a list of all the founded argument matches for this entry ([{COMP: Token}])
		"""

		matches = []

		# Match the arguments based on each subcat for this word entry
		for subcat_type in self.subcats.keys():
			matches += self.subcats[subcat_type].match_arguments(argument_candidates, referenced_token)

		# Match arguments also based on the "next" entry in the lexicon
		# Meaning, the aruguments properties of the same word with another sense
		if self.next is not None:
			matches += self.next.match_arguments(argument_candidates, referenced_token)

		# Sort the matches based on the number of arguments
		matches = sorted(matches, key=lambda k: len(k.keys()), reverse=True)

		# Find only the unique matches, which aren't sub-matches
		unique_matches = []
		for match in matches:
			is_sub_match = any([match.items() <= other_match.items() for other_match in unique_matches])

			if not is_sub_match:
				unique_matches.append(match)

		return unique_matches