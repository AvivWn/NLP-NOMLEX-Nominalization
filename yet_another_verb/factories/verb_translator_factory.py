from argparse import ArgumentParser
from typing import Optional

from yet_another_verb.configuration import VERB_TRANSLATORS_CONFIG
from yet_another_verb.factories.factory import Factory
from yet_another_verb.nomlex.nomlex_version import NomlexVersion
from yet_another_verb.word_to_verb.verb_translator import VerbTranslator
from yet_another_verb.word_to_verb.nomlex_verb_translator import NomlexVerbTranslator


class VerbTranslatorFactory(Factory):
	def __init__(self, translation_mode: str = VERB_TRANSLATORS_CONFIG.TRANSLATOR, **kwargs):
		self.translation_mode = translation_mode
		self.params = kwargs

	def __call__(self) -> VerbTranslator:
		if self.translation_mode == "nomlex":
			return NomlexVerbTranslator(**self.params)

	@staticmethod
	def expand_parser(arg_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
		arg_parser.add_argument(
			"--translation-mode", "-m", choices=["nomlex"], default=VERB_TRANSLATORS_CONFIG.TRANSLATOR,
			help="Defines the method of arguments extraction"
		)
		arg_parser.add_argument(
			"--nomlex-version", type=NomlexVersion, default=VERB_TRANSLATORS_CONFIG.NOMLEX_VERSION,
			help="NOMLEX's lexicon version"
		)
		return arg_parser
