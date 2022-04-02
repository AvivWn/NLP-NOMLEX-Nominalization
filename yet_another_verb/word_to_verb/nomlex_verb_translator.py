import abc
import os
from os.path import join
from typing import Optional

from yet_another_verb.configuration import VERB_TRANSLATORS_CONFIG
from yet_another_verb.file_handlers import JsonFileHandler
from yet_another_verb.file_handlers.file_extensions import JSON_EXTENSION
from yet_another_verb.nomlex.adaptation.entry.entry_division import devide_to_entries
from yet_another_verb.nomlex.adaptation.entry.entry_simplification import simplify_entry
from yet_another_verb.nomlex.constants import EntryProperty
from yet_another_verb.nomlex.nomlex_maestro import NomlexMaestro
from yet_another_verb.word_to_verb.verb_translator import VerbTranslator


class NomlexVerbTranslator(VerbTranslator):
	def __init__(self, nomlex_version: str, **kwargs):
		self.json_entries = NomlexMaestro(nomlex_version).get_json_lexicon()

		dictionary_path = join(
			VERB_TRANSLATORS_CONFIG.TRANSLATORS_CACHE_DIR, "nomlex-verb-translator",
			f"{nomlex_version}.{JSON_EXTENSION}")

		if VERB_TRANSLATORS_CONFIG.USE_CACHE and os.path.exists(dictionary_path):
			self.verb_dictionary = JsonFileHandler.load(dictionary_path)
		else:
			self.verb_dictionary = self._generate_dictionary()
			JsonFileHandler.save(dictionary_path, self.verb_dictionary)

	@staticmethod
	def _add_to_dictionary(verb_dictionary: dict, entry: dict):
		if entry[EntryProperty.TYPE] != 'NOM':
			return

		orth = entry.get(EntryProperty.ORTH)
		verb = entry.get(EntryProperty.VERB)

		if verb is None or len(verb.split()) > 1:
			return

		verb_dictionary[verb] = verb

		if orth is None or len(orth.split()) > 1:
			return

		verb_dictionary[orth] = verb

	def _generate_dictionary(self):
		verb_dictionary = {}
		for raw_entry in self.json_entries:
			simplify_entry(raw_entry)

			for entry in devide_to_entries(raw_entry):
				self._add_to_dictionary(verb_dictionary, entry)

		return verb_dictionary

	def is_transable(self, word: str) -> bool:
		return word in self.verb_dictionary

	def translate(self, word: str) -> Optional[str]:
		return self.verb_dictionary.get(word)
