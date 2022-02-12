from argparse import ArgumentParser
from typing import Optional

from yet_another_verb import ArgsExtractor, NomlexArgsExtractor
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.factories.factory import Factory
from yet_another_verb.configuration import EXTRACTORS_CONFIG
from yet_another_verb.nomlex.nomlex_version import NomlexVersion


class ExtractorFactory(Factory):
	def __init__(self, extraction_mode: str = EXTRACTORS_CONFIG.EXTRACTOR, **kwargs):
		self.extraction_mode = extraction_mode
		self.params = kwargs

	def __call__(self) -> ArgsExtractor:
		if self.extraction_mode == "nomlex":
			dependency_parser = DependencyParserFactory(**self.params)()
			return NomlexArgsExtractor(**self.params, dependency_parser=dependency_parser)

	@staticmethod
	def expand_parser(arg_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
		arg_parser = ExtractorFactory._expand_optional_parser(arg_parser)
		arg_parser.add_argument(
			"--extraction-mode", "-m", choices=["nomlex"], default=EXTRACTORS_CONFIG.EXTRACTOR,
			help="Defines the method of arguments extraction"
		)
		arg_parser.add_argument(
			"--ignore-nomlex-cache", "-c", action="store_true",
			help="Ignores cached nomlex adaptation files instead of recreating them"
		)
		arg_parser.add_argument(
			"--nomlex-version", "-v", type=NomlexVersion, default=EXTRACTORS_CONFIG.NOMLEX_VERSION,
			help="NOMLEX's lexicon version"
		)
		arg_parser = DependencyParserFactory.expand_parser(arg_parser)
		return arg_parser
