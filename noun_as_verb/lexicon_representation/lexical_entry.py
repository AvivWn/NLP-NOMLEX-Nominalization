from collections import defaultdict

from spacy.tokens import Token

from .lexical_subcat import LexicalSubcat
from noun_as_verb.constants.lexicon_constants import *
from noun_as_verb.lisp_to_json.utils import without_part, get_verb_type


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

	def get_next(self):
		return self.next

	def get_subcats(self):
		return self.subcats.values()

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

		if self.next and not self.next.must_be_predicate(word):
			return False

		return True

	# def can_be_noun(self):

