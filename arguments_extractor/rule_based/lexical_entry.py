from collections import defaultdict
from spacy.tokens import Token

from arguments_extractor.rule_based.lexical_subcat import LexicalSubcat
from arguments_extractor.constants.lexicon_constants import *

class Entry:
	orth: str
	subcats = defaultdict(LexicalSubcat)
	next = None
	plural = None		# For noms
	nom = None			# For verbs
	verb = None			# For noms
	singular = None		# For plural noms
	plural_freq = None
	is_verb: bool

	def __init__(self, entry: dict, is_verb):
		self.orth = entry[ENT_ORTH]
		self.nom_type = entry.get(ENT_NOM_TYPE, {})

		self.subcats = defaultdict(LexicalSubcat)
		for subcat_type in entry.get(ENT_VERB_SUBC, {}).keys():
			self.subcats[subcat_type] = LexicalSubcat(entry[ENT_VERB_SUBC][subcat_type], subcat_type, is_verb)

		self.next = entry.get(ENT_NEXT, None)
		self.plural = entry.get(ENT_PLURAL, None)
		self.nom = entry.get(ENT_NOM, None)
		self.verb = entry.get(ENT_VERB, None)
		self.singular = entry.get(ENT_SINGULAR, None)
		self.plural_freq = entry.get(ENT_PLURAL_FREQ, None)
		self.is_verb = is_verb

	def set_next(self, lexicon):
		self.next = lexicon.get_entry(self.next)

	@staticmethod
	def _choose_informative_matches(extractions):
		# Sort the extractions based on the number of founded arguments
		extractions = sorted(extractions, key=lambda extraction: len(extraction.match.keys()), reverse=True)

		# Chose only the relevant extractions (the ones with the maximum number of arguments)
		relevant_extractions = []
		for extraction in extractions:
			match = extraction.match

			if len(match.keys()) != len(extractions[0].match.keys()):
				continue

			is_sub_match = any([other_extraction.is_sub_extraction(extraction) for other_extraction in relevant_extractions])

			if not is_sub_match:
				relevant_extractions.append(extraction)

		#@TODO- does it find the most informative matches?
		informative_extractions = []
		for extraction in relevant_extractions:
			found_more_specific_extraction = False

			for other_extraction in relevant_extractions:
				if extraction.is_more_informative(other_extraction):
					found_more_specific_extraction = True
					break

			if not found_more_specific_extraction:
				informative_extractions.append(extraction)

		return informative_extractions



	def match_arguments(self, argument_candidates: list, referenced_token: Token):
		"""
		Matches the given argument candidates to the possible arguments of all the entries with the same orth (this, next and so on)
		:param argument_candidates: the candidates for the arguments of this entry (as list of tokens)
		:param referenced_token: the predicate of the arguments that we are after
		:return: A list of all the possible argument extractions for this entry ([Extraction])
		"""

		extractions = []

		# Match the arguments based on each subcat for this word entry
		for subcat_type in self.subcats.keys():
			extractions += self.subcats[subcat_type].match_arguments(argument_candidates, referenced_token)

		# Match arguments also based on the "next" entry in the lexicon
		# Meaning, the aruguments properties of the same word with another sense
		if self.next is not None:
			extractions += self.next.match_arguments(argument_candidates, referenced_token)

		extractions = self._choose_informative_matches(extractions)

		return extractions