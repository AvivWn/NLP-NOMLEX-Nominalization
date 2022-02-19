import os

from yet_another_verb.nomlex.lisp_to_json.lisp_to_json import lisps_to_jsons
from yet_another_verb.nomlex.adaptation.lexicon_adaptation import generate_adapted_lexicon
from yet_another_verb.nomlex.representation.lexicon import Lexicon
from yet_another_verb.file_handlers import TXTFileHandler, JsonFileHandler, PKLFileHandler
from yet_another_verb.configuration.extractors_config import EXTRACTORS_CONFIG


class NomlexMaestro:
	def __init__(self, nomlex_version: str = EXTRACTORS_CONFIG.NOMLEX_VERSION):
		self.nomlex_version = nomlex_version

	def get_original_lexicon(self) -> str:
		lisp_path = f"{EXTRACTORS_CONFIG.LISP_LEXICON_DIR}/{self.nomlex_version}.txt"
		file_text = TXTFileHandler(as_lines=False).load(lisp_path)
		return " ".join(file_text.splitlines())

	def get_adapted_json_lexicon(self) -> Lexicon:
		json_path = f"{EXTRACTORS_CONFIG.JSON_LEXICON_DIR}/{self.nomlex_version}.json"
		if EXTRACTORS_CONFIG.USE_NOMLEX_CACHE and os.path.exists(json_path):
			return Lexicon.from_json(JsonFileHandler.load(json_path))

		lisp_lexicon = self.get_original_lexicon()
		json_data = lisps_to_jsons(lisp_lexicon)
		lexicon = generate_adapted_lexicon(json_data)
		# JsonFileHandler.save(json_path, lexicon.to_json())
		return lexicon

	def get_adapted_lexicon(self) -> Lexicon:
		pkl_path = f"{EXTRACTORS_CONFIG.PKL_LEXICON_DIR}/{self.nomlex_version}.pkl"
		if EXTRACTORS_CONFIG.USE_NOMLEX_CACHE and os.path.exists(pkl_path):
			return PKLFileHandler.load(pkl_path)

		lexicon = self.get_adapted_json_lexicon()
		PKLFileHandler.save(pkl_path, lexicon)
		return lexicon
