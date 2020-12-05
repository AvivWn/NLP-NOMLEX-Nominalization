import os
import re
import json
import pickle
import numpy as np
from collections import defaultdict
from os.path import join

from tqdm import tqdm
from spacy.tokens import Token

from noun_as_verb.constants.lexicon_constants import ENT_VERB, ENT_ORTH, NOM_TYPE_NONE, VERB_TYPE_NONE, DEFAULT_ENTRY, ENT_SINGULAR
from noun_as_verb.constants.ud_constants import UPOS_VERB, UPOS_NOUN
from noun_as_verb.utils import get_dependency_tree, get_lexicon_path
from noun_as_verb import config

# Contains pairs of (verb, noun) and can retrieve:
# - the most appropriate verb of a noun (one to one)
# - the possible appropriate nouns of a verb (one to many)
class VerbNounMatcher:
	CATVAR_PATH = join(os.path.dirname(__file__), "catvar21.signed")
	MATCHER_PATH = join(os.path.dirname(__file__), "verb_noun_matcher.pkl")

	# The two dictionaries are generated from all the (verb, noun) pairs, founded in NOMLEX and CATVAR
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

		# self.count_predicates()

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

	def get_possible_types(self, verb):
		verb_types = self.verb_types_per_verb.get(verb, {VERB_TYPE_NONE})
		verb_types = {VERB_TYPE_NONE} if len(verb_types) == 0 else verb_types
		verb_types = sorted(list(verb_types))

		nom_types = self.nom_types_per_verb.get(verb, {NOM_TYPE_NONE})
		nom_types = {NOM_TYPE_NONE} if len(nom_types) == 0 else nom_types
		nom_types = sorted(list(nom_types))

		return verb_types, nom_types



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
			# Avoid the default nom-entry
			if nom_entry[ENT_ORTH] == DEFAULT_ENTRY:
				continue

			# Ignore any entry that refer to a plural form of a nom
			if ENT_SINGULAR in nom_entry:
				continue

			verb, noun = nom_entry[ENT_VERB], nom_entry[ENT_ORTH]
			verb = verb.split("#")[0] # Avoid verb count
			noun = noun.split("#")[0] # Avoid nom count
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
		catvar_pairs = self._get_catvar_pairs()

		joint_pairs = set(nomlex_pairs).intersection(catvar_pairs)
		total_pairs = set(nomlex_pairs + catvar_pairs)
		print(f"The number of (verb, noun) pairs: NOMLEX={len(nomlex_pairs)}, CATVAR={len(catvar_pairs)}, joint={len(joint_pairs)}")

		# Creates the dictionaries from the founded pairs
		verbs_per_noun = defaultdict(set)
		for verb, noun in total_pairs:
			verbs_per_noun[noun].add(verb)
			self.nouns_per_verb[verb].add(noun)

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

		# unique_nom_types_sets = set(frozenset(s) for s in self.nom_types_per_verb.values())
		# unique_verb_types_sets = set(frozenset(s) for s in self.verb_types_per_verb.values())

	def count_predicates(self):
		verbs_count = 0
		noms_count = 0

		# Creates the dictionaries from the NOMLEX pairs
		nomlex_pairs = self._get_nomlex_pairs(get_lexicon_path(config.NOMLEX_PLUS_NAME, "json"))
		verbs = [pair[0] for pair in nomlex_pairs]
		noms = [pair[1] for pair in nomlex_pairs]

		# Counts the number of verbs appearances in a random dataset
		limited_n_sentences = 10 ** 5  # 10 ** 6
		sentences_file = open(config.WIKI_SENTENCES_PATH)
		pbar = tqdm(enumerate(sentences_file), leave=False)
		for i, sentence in pbar:
			doc = get_dependency_tree(sentence, disable=['ner', 'parser'])

			for word in doc:
				if word.pos_ == UPOS_VERB and word.lemma_ in verbs:
					verbs_count +=1
				elif word.pos_ == UPOS_NOUN and word.lemma_ in noms:
					noms_count += 1

			pbar.set_description(f"VERBS={verbs_count}, NOMS={noms_count}")

			if i > limited_n_sentences:
				break

		print(f"Founded {verbs_count} verbs")
		print(f"Founded {noms_count} noms")