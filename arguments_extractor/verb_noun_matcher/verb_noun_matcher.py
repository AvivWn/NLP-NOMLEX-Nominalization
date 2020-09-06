import os
import re
import json
import pickle
import numpy as np
from collections import defaultdict
from os.path import join

from tqdm import tqdm
from spacy.tokens import Token

from arguments_extractor.constants.lexicon_constants import ENT_VERB, ENT_ORTH
from arguments_extractor.constants.ud_constants import UPOS_VERB
from arguments_extractor.utils import get_dependency_tree, get_lexicon_path
from arguments_extractor import config

# Contains pairs of (verb, noun) and can retrieve:
# - the most appropriate verb of a noun (one to one)
# - the possible appropriate nouns of a verb (one to many)
class VerbNounMatcher:
	CATVAR_PATH = join(os.path.dirname(__file__), "catvar21.signed")
	MATCHER_PATH = join(os.path.dirname(__file__), "verb_noun_matcher.pkl")

	# The two dictionaries are generated from all the (verb, noun) pairs, founded in
	# They don't contain plural nouns, only singular
	verb_per_noun: dict
	nouns_per_verb: dict

	# Possible nom and verb types per verb
	nom_types_per_verb: dict
	verb_types_per_verb: dict

	def __init__(self, verb_lexicon, nom_lexicon):
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
			self.find_possible_types()

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
		:param word: spacy token of a word
		:return: the suitable verb
		"""

		lemma = word.lemma_.lower()

		if word.pos_ == UPOS_VERB:
			return lemma if lemma in self.nouns_per_verb else None
		else:
			return self.verb_per_noun.get(lemma, None)

	def get_suitable_nouns(self, verb):
		return self.nouns_per_verb.get(verb, set())

	def get_all_forms(self, verb):
		word_forms = self.get_suitable_nouns(verb)
		word_forms.add(verb)
		return word_forms



	def _get_catvar_pairs(self):
		with open(self.CATVAR_PATH, "r") as catvar_db_file:
			catvar_lines = catvar_db_file.readlines()

		pairs = []
		for line in catvar_lines:
			line = line.strip(" \r\n\t")
			line = re.sub('\d', '', line)
			family_words = line.split("#")
			nouns = []
			verbs = []

			# Moving over the word family in the current line
			for word in family_words:
				# Aggregating nouns of the family
				if word.endswith("_N%"):
					nouns.append(word.split("_")[0])

				# Aggregating verbs of the family
				elif word.endswith("_V%"):
					verbs.append(word.split("_")[0])

			# Adding all the founded verbs with the founded nouns, in the current line
			for verb in verbs:
				for noun in nouns:
					pairs.append((verb, noun))

		return pairs

	@staticmethod
	def _get_nomlex_pairs(nomlex_path):
		with open(nomlex_path, "r") as nomlex_file:
			nomlex_lexicon = json.load(nomlex_file)

		pairs = []
		for nom_entry in nomlex_lexicon.values():
			verb, noun = nom_entry[ENT_VERB], nom_entry[ENT_ORTH]
			pairs.append((verb, noun))

		return pairs

	@staticmethod
	def _count_verbs_appearances(verbs: set):
		verbs_appearances = defaultdict(int)

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
		nomlex_pairs = self._get_nomlex_pairs(get_lexicon_path(config.NOMLEX_PLUS_NAME, "json"))
		catvar_pairs = self._get_catvar_pairs()
		joint_pairs = set(nomlex_pairs).intersection(catvar_pairs)
		total_pairs = set(nomlex_pairs + catvar_pairs)
		print(f"The number of (verb, noun) pairs: NOMLEX={len(nomlex_pairs)}, CATVAR={len(catvar_pairs)}, joint={len(joint_pairs)}")

		# Creates the dictionaries from the founded pairs
		verbs_per_noun = defaultdict(set)
		for verb, noun in total_pairs:
			verbs_per_noun[noun].update([verb])
			self.nouns_per_verb[verb].update([noun])

		# For each noun, choose the most common verb
		verbs_appearances = self._count_verbs_appearances(set(self.nouns_per_verb.keys()))
		for noun, verbs in verbs_per_noun.items():
			verbs_list = list(verbs)
			counts = [verbs_appearances[verb] for verb in verbs]
			self.verb_per_noun[noun] = verbs_list[int(np.argmax(counts))]

		print(f"The number of estimated noms: {len(self.verb_per_noun.keys())}")



	def find_possible_types(self):
		"""
		Finds all the possible verbal and nominal types for each verb
		"""

		for verb, nouns in self.nouns_per_verb.items():
			verb_entry = self.verb_lexicon.get_entry(verb)
			if verb_entry is None:
				continue

			verb_types = verb_entry.get_verb_types()
			self.verb_types_per_verb[verb].update(verb_types)

			for noun in nouns:
				nom_entry = self.nom_lexicon.get_entry(noun)
				if nom_entry is None:
					continue

				nom_types = nom_entry.get_nom_types()
				self.nom_types_per_verb[verb].update(nom_types)