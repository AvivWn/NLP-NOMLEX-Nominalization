import os
import json
import random
import pickle
from collections import defaultdict

import numpy as np
from spacy.tokens import Token
from tqdm import tqdm

from arguments_extractor.rule_based.lexical_entry import Entry
from arguments_extractor.rule_based.utils import get_argument_candidates
from arguments_extractor.utils import get_lexicon_path
from arguments_extractor.constants.ud_constants import *
from arguments_extractor import config

class Lexicon:
	entries = defaultdict(Entry)
	is_verb: bool

	MAX_N_CANDIDATES = 5

	def __init__(self, lisp_file_name, is_verb=False):
		json_file_path = get_lexicon_path(lisp_file_name, "json", is_verb=is_verb, is_nom=not is_verb)
		pkl_file_path = get_lexicon_path(lisp_file_name, "pkl", is_verb=is_verb, is_nom=not is_verb)

		# Should we create the Lexicon classes again (pkl formated lexicon)?
		if config.LOAD_LEXICON and os.path.exists(pkl_file_path):
			with open(pkl_file_path, "rb") as input_file:
				loaded_lexicon = pickle.load(input_file)
				self.use_loaded_lexicon(loaded_lexicon)
		else:
			self.entries = defaultdict(Entry)
			self.is_verb = is_verb

			# Loading the lexicon from the json file with the given name
			with open(json_file_path, "r") as input_file:
				lexicon_json = json.load(input_file)

			# Adding each entry of the lexicon to this object
			for entry_word in tqdm(lexicon_json.keys(), "Modeling the lexicon", leave=False):
				self.entries[entry_word] = Entry(lexicon_json[entry_word], is_verb)

			# Update the next entry for the linked entry
			for entry_word in self.entries.keys():
				self.entries[entry_word].set_next(self)

			with open(pkl_file_path, 'wb') as output_file:
				pickle.dump(self, output_file)

	def get_entry(self, entry_word):
		if entry_word == "" or entry_word is None:
			return None

		if entry_word not in self.entries.keys():
			raise Exception(f"The word {entry_word} do not appear in this lexicon!")

		return self.entries[entry_word]

	def get_entries(self):
		return self.entries

	def is_verbal_lexicon(self):
		return self.is_verb

	def use_loaded_lexicon(self, loaded_lexicon):
		self.entries = loaded_lexicon.get_entries()
		self.is_verb = loaded_lexicon.is_verbal_lexicon()

	def split_noms(self, ratio):
		if self.is_verb:
			return []

		all_noms = [entry.orth for entry in self.entries.values() if entry.nom is None]
		all_noms = np.unique(all_noms)

		random.shuffle(all_noms)

		train_part = int(len(all_noms) * ratio)
		train_limited_noms = all_noms[:train_part]
		test_limited_noms = all_noms[train_part:]

		return train_limited_noms, test_limited_noms

	def find(self, word: Token):
		"""
		Finds the given word in this lexicon
		:param word: word token (spacy token)
		:return: the suitable word in the lexicon, or None otherwise
		"""

		if self.is_verb and word.pos_ != UPOS_VERB:
			return None

		if not self.is_verb and word.pos_ != UPOS_NOUN:
			return None

		word_forms = [word.orth_, word.orth_.lower(), word.lemma_]

		for word_form in word_forms:
			if word_form in self.entries.keys():
				word_entry = self.entries[word_form]

				return word_entry

		return None



	def extract_arguments(self, dependency_tree: list, min_arguments=0):
		"""
		Extracts the arguments for any relevant word of the given sentence that appear in this lexicon
		:param dependency_tree: the appropriate dependency tree for a sentence
		:param min_arguments: The minimum number of arguments for any founed extraction (0 is deafult)
		:return: all the founded argument extractions for any relevant word ({Token: [{COMP: Span}]})
		"""

		extractions_per_word = defaultdict(list)

		for token in dependency_tree:
			# Ignore words that don't appear in this lexicon
			word_entry = self.find(token)
			if word_entry is None:
				continue

			lexical_word = word_entry.orth

			# Get the candidates for the arguments of this word (based relevant direct links in the ud)
			argument_candidates = get_argument_candidates(token, self.is_verb)

			if len(argument_candidates) > self.MAX_N_CANDIDATES:
				continue

			# if config.DEBUG: print(f"Candidates for {token.orth_}:", [candidate_token._.subtree_text if candidate_token != token else candidate_token.orth_ for candidate_token in argument_candidates])

			# Get all the possible extractions of this word
			extractions = self.entries[lexical_word].match_arguments(argument_candidates, token)

			for extraction in extractions:
				if len(extraction.get_complements()) >= min_arguments:
					extractions_per_word[token].append(extraction.as_dict())

			if config.DEBUG and len(extractions_per_word.get(token, [])) > 1:
				pass
				# print(extractions_per_word[token])
				# raise Exception("Found word with more than one legal extraction.")

		return extractions_per_word