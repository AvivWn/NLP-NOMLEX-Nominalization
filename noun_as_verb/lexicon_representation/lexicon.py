import os
import json
import pickle
from collections import defaultdict

from spacy.tokens import Token
from tqdm import tqdm

from .lexical_entry import Entry
from .utils import is_noun, is_verb
from noun_as_verb.lisp_to_json.lisp_to_json import lisp_to_json
from noun_as_verb.utils import get_lexicon_path
from noun_as_verb import config


class Lexicon:
	entries = defaultdict(Entry)
	is_verb: bool

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

			if not os.path.exists(json_file_path):
				lisp_to_json(lisp_file_name)

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

		return self.entries.get(entry_word, None)

	def get_entries(self):
		return self.entries

	def is_verbal_lexicon(self):
		return self.is_verb

	def use_loaded_lexicon(self, loaded_lexicon):
		self.entries = loaded_lexicon.get_entries()
		self.is_verb = loaded_lexicon.is_verbal_lexicon()

	def get_verbs_by_arg(self, complement_type):
		# Finds the verbs that can be complemented by the given complement type

		suitable_verbs = set()
		for entry in self.entries.values():
			if entry.is_default_entry():
				continue

			suitable_verb = entry.get_suitable_verb()

			# The complement type is the nom
			if complement_type in entry.get_nom_types(ignore_part=True):
				suitable_verbs.update([suitable_verb])

			# The argument of that type can complement the verb\nom
			for subcat in entry.get_subcats():
				if complement_type in subcat.get_args_types():
					suitable_verbs.update([suitable_verb])
					break

		return list(suitable_verbs)

	def is_contain(self, word: Token):
		return self.find(word) != []

	def find(self, word: Token):
		if self.is_verb and is_noun(word):
			return []

		if not self.is_verb and is_verb(word):
			return []

		word_form = word.lemma_.lower() if self.is_verb else word.orth_.lower()
		word_entry = self.entries.get(word_form, None)

		if not word_entry:
			return []

		word_entries = [word_entry]
		while word_entry:
			word_entries.append(word_entry)
			word_entry = word_entry.get_next()

		return word_entries
