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
		self.orth = entry.get(ENT_ORTH, {})
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

	def _choose_informative_matches(self, extractions: list, referenced_token: Token, suitable_verb: str, arguments_predictor=None):
		# Sort the extractions based on the number of founded arguments
		extractions.sort(key=lambda extraction: len(extraction.get_complements()), reverse=True)

		# Chose only the relevant extractions (the ones with the maximum number of arguments)
		relevant_extractions = []
		args_per_candidates = defaultdict(list)
		possible_matches = []
		for extraction in extractions:
			possible_matches.append(extraction.as_properties_dict())

			if len(extraction.get_complements()) != len(extractions[0].get_complements()):
				continue

			is_sub_match = any([other_extraction.is_sub_extraction(extraction) for other_extraction in relevant_extractions])

			if is_sub_match:
				continue

			relevant_extractions.append(extraction)

			# Aggregate all the possible complement types per candidate
			for argument in extraction.match.values():
				if argument not in args_per_candidates[argument.argument_token]:
					args_per_candidates[argument.argument_token].append(argument)

		# Determine the complement type of candidates with uncertainty about their complement type
		if arguments_predictor is not None and self.orth != DEFAULT_ENTRY:
			args_per_candidates = arguments_predictor.determine_args_type(args_per_candidates, referenced_token, suitable_verb)

			# Filter out potentially wrong arguments within extractions
			filtered_extractions = []
			for extraction in relevant_extractions:
				# Create a new matching of arguments and complement types by removing inapropriate pairs
				new_match = {}
				for complement_type, argument in extraction.match.items():
					if argument in args_per_candidates[argument.argument_token]:
						new_match[complement_type] = argument

				# Did the new match appear as a possible match at first place?
				extraction.match = new_match
				if extraction.as_properties_dict() not in possible_matches:
					continue

				filtered_extractions.append(extraction)

			filtered_extractions.sort(key=lambda extraction: len(extraction.get_complements()), reverse=True)
			relevant_extractions = [extraction for extraction in filtered_extractions if len(extraction.get_complements()) == len(filtered_extractions[0].get_complements())]

		#@TODO- does it find the most informative matches?
		informative_extractions = []
		for extraction in relevant_extractions:
			found_more_specific_extraction = False

			# Check if there are more informative extractions than this one
			for other_extraction in relevant_extractions:
				if extraction.is_more_informative(other_extraction):
					found_more_specific_extraction = True
					break

			if not found_more_specific_extraction:
				informative_extractions.append(extraction)

		return informative_extractions



	def match_arguments(self, argument_candidates: list, referenced_token: Token, suitable_verb: str, arguments_predictor=None):
		"""
		Matches the given argument candidates to the possible arguments of all the entries with the same orth (this, next and so on)
		:param argument_candidates: the candidates for the arguments of this entry (as list of tokens)
		:param referenced_token: the predicate of the arguments that we are after
		:param suitable_verb: the appropriate verb for the given reference token
		:param arguments_predictor: the model-based extractor object to determine the argument type of a span (optional)
		:return: A list of all the possible argument extractions for this entry ([Extraction])
		"""

		extractions = []

		# Match the arguments based on each subcat for this word entry
		for subcat_type in self.subcats.keys():
			extractions += self.subcats[subcat_type].match_arguments(argument_candidates, referenced_token, suitable_verb, arguments_predictor=arguments_predictor)

		# Match arguments also based on the "next" entry in the lexicon
		# Meaning, the aruguments properties of the same word with another sense
		if self.next is not None:
			extractions += self.next.match_arguments(argument_candidates, referenced_token, suitable_verb, arguments_predictor=arguments_predictor)

		extractions = self._choose_informative_matches(extractions, referenced_token, suitable_verb, arguments_predictor=arguments_predictor)

		return extractions