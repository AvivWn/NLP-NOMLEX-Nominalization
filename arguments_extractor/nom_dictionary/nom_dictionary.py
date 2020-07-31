import os
import re
import json
import pickle

from arguments_extractor.constants.lexicon_constants import ENT_VERB, ENT_ORTH, ENT_PLURAL, NONE_VALUES
from arguments_extractor import config

class NomDictionary:
	CATVAR_PATH = os.path.join(os.path.dirname(__file__), "catvar21.signed")
	DICTIONARY_PATH = os.path.join(os.path.dirname(__file__), "nom_dictionary.pkl")

	verb_per_nom: dict

	def __init__(self):
		self.verb_per_nom = {}

		if config.LOAD_LEXICON and os.path.exists(self.DICTIONARY_PATH):
			self.load_dictionary()
		else:
			self.generate_dict()

			with open(self.DICTIONARY_PATH, "wb") as dictionary_file:
				pickle.dump(self, dictionary_file)

	def get_suitable_verb(self, possible_nom):
		if possible_nom in self.verb_per_nom.keys():
			return self.verb_per_nom[possible_nom]

		return None

	def load_dictionary(self):
		with open(self.DICTIONARY_PATH, "rb") as dictionary_file:
			dictionary_object = pickle.load(dictionary_file)
			self.verb_per_nom = dictionary_object.verb_per_nom

	def update_by_catvar(self):
		# Moving over the catvar database file
		# Finding the verb and nouns in the same line, meaning in the same words family
		with open(self.CATVAR_PATH, "r") as catvar_db_file:
			catvar_lines = catvar_db_file.readlines()

		for line in catvar_lines:
			line = line.strip(" \r\n\t")
			line = re.sub('\d', '', line)
			family_words = line.split("#")
			nouns = []
			verbs = []

			# Moving over the words in the current line
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
					self.verb_per_nom[noun] = verb

	def update_by_nomlex(self, nomlex_path):
		with open(nomlex_path, "r") as nomlex_file:
			nomlex_lexicon = json.load(nomlex_file)

		for nom_entry in nomlex_lexicon.values():
			self.verb_per_nom[nom_entry[ENT_ORTH]] = nom_entry[ENT_VERB]

			# Add also the plural of the nominalization if it is existed
			plural = nom_entry.get(ENT_PLURAL, None)
			if plural not in NONE_VALUES + [None]:
				self.verb_per_nom[plural] = nom_entry[ENT_VERB]

	def generate_dict(self):
		self.update_by_catvar()
		self.update_by_nomlex(config.JSON_DIR + "NOMLEX-2001.json")
		self.update_by_nomlex(config.JSON_DIR + "NOMLEX-plus.1.0.json")