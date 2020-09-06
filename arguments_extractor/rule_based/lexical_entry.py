from collections import defaultdict

from spacy.tokens import Token

from arguments_extractor.rule_based.lexical_subcat import LexicalSubcat
from arguments_extractor.constants.lexicon_constants import *
from arguments_extractor.lisp_to_json.utils import without_part, get_verb_type
from arguments_extractor.utils import filter_list, aggregate_to_dict

class Entry:
	orth: str
	subcats = defaultdict(LexicalSubcat)
	next = None
	plural = None		# For noms
	nom = None			# For verbs
	verb = None			# For noms
	singular = None		# For plural noms
	plural_freq = None
	singular_false = False
	noun_properties = []

	is_verb: bool

	def __init__(self, entry: dict, is_verb):
		self.orth = entry.get(ENT_ORTH, "")
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
		self.singular_false = entry.get(ENT_SINGULAR_FALSE, False)
		self.noun_properties = entry.get(ENT_NOUN, [])
		self.is_verb = is_verb

	def set_next(self, lexicon):
		self.next = lexicon.get_entry(self.next)

	def get_suitable_verb(self):
		# Returns the suitable verb for this entry
		# Assumption- every nominalization in nomlex is appropriate to exactly one verb
		# This is true even if it has multiple entries

		if self.is_verb:
			suitable_verb = self.orth
		else:  # Nominalization entry
			suitable_verb = self.verb

		return suitable_verb.split("#")[0]

	def get_nom_types(self, ignore_part=False):
		# Returns all the possible types for the nom in this entry

		if self.is_verb:
			return []

		entry = self
		nom_types = set()

		# Include also the other entries that appropriate for the same nom
		while entry is not None:
			nom_type = self.nom_type[TYPE_OF_NOM]
			nom_type = without_part(nom_type) if ignore_part else nom_type
			nom_types.add(nom_type)
			entry = entry.next

		return list(nom_types)

	def get_verb_types(self):
		# Returns all the possible types for the verb in this entry

		if not self.is_verb:
			return []

		entry = self
		verb_types = set()

		# Include also the other entries that appropriate for the same nom
		while entry is not None:
			for subcat in self.subcats.keys():
				verb_type = get_verb_type(subcat)
				verb_types.add(verb_type)

			entry = entry.next

		return list(verb_types)

	def is_single_type(self, ignore_part=False):
		# Returns whether the nom in this entry has more than one type in NOMLEX
		# The PART prefix may not be taken into account

		nom_types = self.get_nom_types(ignore_part=ignore_part)
		return len(nom_types) <= 1

	def is_default_entry(self):
		return self.orth == DEFAULT_ENTRY

	def must_be_predicate(self, word: Token, be_certain=True):
		# Returns whether the given word must be a predicate
		# Verb is always verb, but some nominalization can appear as common noun

		if not be_certain or self.is_verb:
			return True

		# Out-of-NOMLEX nouns can always be common nouns
		if self.is_default_entry():
			return False

		if not {NOUN_EXISTS, NOUN_RARE_NOUN, NOUN_RARE_NOM}.isdisjoint(self.noun_properties):
			return False

		# The word is plural, and noun can appear only as plural
		if word.orth_ != word.lemma_ and NOUN_PLUR_ONLY in self.noun_properties:
			return False

		# The word is singular, and noun can appear only as singular
		if word.orth_ == word.lemma and NOUN_SING_ONLY in self.noun_properties:
			return False

		if self.next is not None and self.next.can_be_noun(word):
			return False

		return True



	# def _choose_informative_matches(self, extractions: list, referenced_token: Token, suitable_verb: str, arguments_predictor=None):
	# 	# Sort the extractions based on the number of founded arguments
	# 	extractions.sort(key=lambda extraction: len(extraction.get_complements()), reverse=True)
	#
	# 	# Choose only the relevant extractions (the ones with the maximum number of arguments)
	# 	relevant_extractions = []
	# 	args_per_candidates = defaultdict(list)
	# 	possible_matches = []
	# 	for extraction in extractions:
	# 		possible_matches.append(extraction.as_properties_dict())
	#
	# 		if len(extraction.get_complements()) != len(extractions[0].get_complements()):
	# 			continue
	#
	# 		is_sub_match = any([other_extraction.is_sub_extraction(extraction) for other_extraction in relevant_extractions])
	#
	# 		if is_sub_match:
	# 			continue
	#
	# 		relevant_extractions.append(extraction)
	#
	# 		# Aggregate all the possible complement types per candidate
	# 		for argument in extraction.match.values():
	# 			if argument not in args_per_candidates[argument.argument_token]:
	# 				args_per_candidates[argument.argument_token].append(argument)
	#
	# 	# Determine the complement type of candidates with uncertainty about their complement type
	# 	if arguments_predictor is not None and self.orth != DEFAULT_ENTRY:
	# 		args_per_candidates = arguments_predictor.determine_args_type(args_per_candidates, referenced_token, suitable_verb)
	#
	# 		# Filter out potentially wrong arguments within extractions
	# 		filtered_extractions = []
	# 		for extraction in relevant_extractions:
	# 			# Create a new matching of arguments and complement types by removing inapropriate pairs
	# 			new_match = {}
	# 			for complement_type, argument in extraction.match.items():
	# 				if argument in args_per_candidates[argument.argument_token]:
	# 					new_match[complement_type] = argument
	#
	# 			# Did the new match appear as a possible match at first place?
	# 			extraction.match = new_match
	# 			if extraction.as_properties_dict() not in possible_matches:
	# 				continue
	#
	# 			filtered_extractions.append(extraction)
	#
	# 		filtered_extractions.sort(key=lambda extraction: len(extraction.get_complements()), reverse=True)
	# 		relevant_extractions = [extraction for extraction in filtered_extractions if len(extraction.get_complements()) == len(filtered_extractions[0].get_complements())]
	#
	# 	#@TODO- does it find the most informative matches?
	# 	informative_extractions = []
	# 	for extraction in relevant_extractions:
	# 		found_more_specific_extraction = False
	#
	# 		# Check if there are more informative extractions than this one
	# 		for other_extraction in relevant_extractions:
	# 			if extraction != other_extraction and extraction.is_more_informative(other_extraction):
	# 				found_more_specific_extraction = True
	# 				break
	#
	# 		if not found_more_specific_extraction:
	# 			informative_extractions.append(extraction)
	#
	# 	return informative_extractions

	def _choose_informative(self, extractions: list, referenced_token: Token, suitable_verb: str, arguments_predictor=None):
		# Choose the extractions with max arguments
		all_extractions = extractions
		filter_extractions = lambda extractions: filter_list(extractions, lambda e1,e2: e2.is_sub_extraction(e1), lambda e: e.get_complements(), greedy=True)
		extractions = filter_extractions(extractions)

		# Should we handle uncertainty?
		if arguments_predictor and not self.is_default_entry():
			# Determine the complement type of uncertain candidates, with a model
			candidates_args = aggregate_to_dict([e.get_candidates_args() for e in extractions])
			candidates_args = arguments_predictor.determine_args_type(candidates_args, referenced_token, suitable_verb)

			# Clean extractions from arguments that weren't chosen
			extractions = [e for e in extractions if e.get_filtered(candidates_args).isin(all_extractions)]
			extractions = filter_extractions(extractions)

		# Choose only the informative extractions
		extractions = filter_list(extractions, lambda e1,e2: e1!=e2 and e1.is_more_informative(e2), lambda e: e.get_complements())
		return extractions

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
			extractions += self.subcats[subcat_type].match_arguments(argument_candidates, referenced_token, suitable_verb, arguments_predictor)

		# Match arguments also based on the "next" entry in the lexicon
		# Meaning, the aruguments properties of the same word with another sense
		if self.next is not None:
			extractions += self.next.match_arguments(argument_candidates, referenced_token, suitable_verb, arguments_predictor)

		# extractions = self._choose_informative_matches(extractions, referenced_token, suitable_verb, arguments_predictor)
		extractions = self._choose_informative(extractions, referenced_token, suitable_verb, arguments_predictor)

		return extractions