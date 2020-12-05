import os
import json
import pickle
import numpy as np
from collections import defaultdict
from os.path import join

from tqdm import tqdm
from spacy.tokens import Token

from .verb_noun_translator import VerbNounTranslator
from noun_as_verb.lexicon_representation.utils import is_verb, is_noun
from noun_as_verb.constants.lexicon_constants import ENT_VERB, ENT_ORTH, DEFAULT_ENTRY, ENT_SINGULAR
from noun_as_verb.constants.ud_constants import UPOS_VERB
from noun_as_verb.utils import get_dependency_tree, get_lexicon_path
from noun_as_verb import config


class NomlexTranslator(VerbNounTranslator):
	CATVAR_PATH = join(os.path.dirname(__file__), "catvar21.signed")
	MATCHER_PATH = join(os.path.dirname(__file__), "verb_noun_matcher.pkl")

	# The two dictionaries are generated from all the (verb, noun) pairs, founded in NOMLEX and CATVAR
	# They don't contain plural nouns, only singular
	verb_per_noun: dict
	nouns_per_verb: dict

	def __init__(self, verb_lexicon, nom_lexicon):
		super().__init__()
		self.verb_lexicon = verb_lexicon
		self.nom_lexicon = nom_lexicon

		self.verb_per_noun = {}
		self.nouns_per_verb = defaultdict(set)
		self.nom_types_per_verb = defaultdict(set)
		self.verb_types_per_verb = defaultdict(set)

		if config.LOAD_LEXICON and os.path.exists(self.MATCHER_PATH):
			self.load_dictionary()
		else:
			self.generate_mapping()

			with open(self.MATCHER_PATH, "wb") as dictionary_file:
				pickle.dump(self, dictionary_file)

	def load_dictionary(self):
		with open(self.MATCHER_PATH, "rb") as dictionary_file:
			dictionary_object = pickle.load(dictionary_file)
			self.verb_per_noun = dictionary_object.verb_per_noun
			self.nouns_per_verb = dictionary_object.nouns_per_verb
			self.nom_types_per_verb = dictionary_object.nom_types_per_verb
			self.verb_types_per_verb = dictionary_object.verb_types_per_verb

	def get_suitable_verb(self, word: Token):
		"""
		Finds the appropriate verb of the given word
		None is returned if such verb don't exist in this database
		Assumption- every noun has exactly one suitable verb
		:param word: spacy token of a word
		:return: the suitable verb
		"""

		lemma = word.lemma_.lower()

		if is_verb(word):
			return lemma if lemma in self.nouns_per_verb else None
		elif is_noun(word):
			return self.verb_per_noun.get(lemma, None)

	def get_suitable_nouns(self, verb):
		return self.nouns_per_verb.get(verb, set())

	@staticmethod
	def _get_nomlex_pairs(nomlex_path):
		with open(nomlex_path, "r") as nomlex_file:
			nomlex_lexicon = json.load(nomlex_file)

		pairs = []
		for nom_entry in nomlex_lexicon.values():
			# Avoid the default nom-entry
			if nom_entry[ENT_ORTH] == DEFAULT_ENTRY:
				continue

			# Ignore any entry that refer to a plural form of a nom
			if ENT_SINGULAR in nom_entry:
				continue

			verb, noun = nom_entry[ENT_VERB], nom_entry[ENT_ORTH]
			verb = verb.split("#")[0]  # Avoid verb count
			noun = noun.split("#")[0]  # Avoid nom count
			pairs.append((verb, noun))

		return pairs

	@staticmethod
	def _count_verbs_appearances(verbs: set):
		verbs_appearances = defaultdict(int)

		# Counts the number of verbs appearances in a random dataset
		limited_n_sentences = 10 ** 3 # 10 ** 6
		sentences_file = open(config.WIKI_SENTENCES_PATH)
		for i, sentence in tqdm(enumerate(sentences_file), leave=False):
			doc = get_dependency_tree(sentence, disable=['ner', 'parser'])
			verbs_lemmas = list([w.lemma_ for w in doc if w.pos_ == UPOS_VERB])

			for verb in verbs_lemmas:
				if verb in verbs:
					verbs_appearances[verb] += 1

			if i > limited_n_sentences:
				break

		print(f"Founded {len(verbs_appearances.keys())} verbs from {len(verbs)}")

		return verbs_appearances

	def generate_mapping(self):
		"""
		Generates the mapping of verb <-> noun using pairs founded in NOMLEX and CATVAR
		"""

		# Generates first all the (verb, noun) pairs that are written in NOMLEX + CATVAR
		nomlex_pairs = self._get_nomlex_pairs(get_lexicon_path(config.NOMLEX_PLUS_NAME, "json")) # Using the original nomlex (which includes entries that were deleted from the new one)
		nomlex_pairs += self._get_nomlex_pairs(get_lexicon_path(config.NOMLEX_PLUS_NAME, "json", is_nom=True)) # Using the new adaptation (which includes additional new duplicated entries)
		nomlex_pairs = list(set(nomlex_pairs))

		# Creates the dictionaries from the founded pairs
		verbs_per_noun = defaultdict(set)
		for verb, noun in nomlex_pairs:
			verbs_per_noun[noun].add(verb)
			self.nouns_per_verb[verb].add(noun)

		# For each noun, choose the most common verb
		verbs_appearances = self._count_verbs_appearances(set(self.nouns_per_verb.keys()))
		for noun, verbs in verbs_per_noun.items():
			verbs_list = list(verbs)
			counts = [verbs_appearances[verb] for verb in verbs]
			self.verb_per_noun[noun] = verbs_list[int(np.argmax(counts))]

		print(f"The number of estimated noms: {len(self.verb_per_noun.keys())}")
