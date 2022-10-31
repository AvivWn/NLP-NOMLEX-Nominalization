import os
from os.path import join
from typing import Optional, Set, Union

from yet_another_verb.configuration import VERB_TRANSLATORS_CONFIG
from yet_another_verb.data_handling import PKLFileHandler
from yet_another_verb.data_handling.file.file_extensions import PICKLE_EXTENSION
from yet_another_verb.dependency_parsing import POSTag, POSTaggedWord
from yet_another_verb.nomlex.adaptation.entry.entry_division import devide_to_entries
from yet_another_verb.nomlex.adaptation.entry.entry_simplification import simplify_entry
from yet_another_verb.nomlex.constants import EntryProperty
from yet_another_verb.nomlex.nomlex_maestro import NomlexMaestro
from yet_another_verb.word_to_verb.verb_translator import VerbTranslator


class NomlexVerbTranslator(VerbTranslator):
	def __init__(self, translator_nomlex_version: str = VERB_TRANSLATORS_CONFIG.NOMLEX_VERSION, **kwargs):
		self.json_entries = NomlexMaestro(translator_nomlex_version).get_json_lexicon()

		dictionary_path = join(
			VERB_TRANSLATORS_CONFIG.TRANSLATORS_CACHE_DIR, "nomlex-verb-translator",
			f"{translator_nomlex_version}.{PICKLE_EXTENSION}")

		if VERB_TRANSLATORS_CONFIG.USE_CACHE and os.path.exists(dictionary_path):
			self.verb_dictionary = PKLFileHandler.load(dictionary_path)
		else:
			self.verb_dictionary = self._generate_dictionary()
			PKLFileHandler.save(dictionary_path, self.verb_dictionary)

	@staticmethod
	def _add_to_dictionary(verb_dictionary: dict, entry: dict):
		if entry[EntryProperty.TYPE] != 'NOM':
			return

		orth = entry.get(EntryProperty.ORTH)
		verb = entry.get(EntryProperty.VERB)

		if verb is None or len(verb.split()) > 1:
			return

		verb_dictionary[POSTaggedWord(verb, POSTag.VERB)] = verb

		if orth is None or len(orth.split()) > 1:
			return

		verb_dictionary[POSTaggedWord(orth, POSTag.NOUN)] = verb

	def _generate_dictionary(self):
		verb_dictionary = {}
		for raw_entry in self.json_entries:
			simplify_entry(raw_entry)

			for entry in devide_to_entries(raw_entry):
				self._add_to_dictionary(verb_dictionary, entry)

		return verb_dictionary

	def is_transable(self, word: str, postag: Union[POSTag, str]) -> bool:
		return POSTaggedWord(word, postag) in self.verb_dictionary

	def translate(self, word: str, postag: Union[POSTag, str]) -> Optional[str]:
		return self.verb_dictionary.get(POSTaggedWord(word, postag))

	@property
	def supported_words(self) -> Set[POSTaggedWord]:
		return set(self.verb_dictionary.keys())

	@property
	def verbs(self) -> Set[str]:
		return set(self.verb_dictionary.values())
