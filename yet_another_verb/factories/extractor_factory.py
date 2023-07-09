from argparse import ArgumentParser
from typing import Optional

from yet_another_verb import ArgsExtractor
from yet_another_verb.arguments_extractor.extractors.verb_references_based.verb_references.utils import \
	load_extraction_references, get_references_by_predicate
from yet_another_verb.configuration.extractors_config import EXTRACTOR_BY_NAME, EXTRACTORS_CONFIG, \
	CONSTRAINTS_EXTRACTORS, VERB_REFERENCES_EXTRACTORS
from yet_another_verb.data_handling import ExtractedFileHandler
from yet_another_verb.factories.argument_encoder_factory import ArgumentEncoderFactory
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.factories.factory import Factory
from yet_another_verb.factories.verb_translator_factory import VerbTranslatorFactory
from yet_another_verb.nomlex.nomlex_version import NomlexVersion
from yet_another_verb.sentence_encoding.argument_encoding.utils import arg_encoder_to_tuple_id


class ExtractorFactory(Factory):
	def __init__(self, extraction_mode: str = EXTRACTORS_CONFIG.EXTRACTOR, **kwargs):
		self.extraction_mode = extraction_mode
		self.params = kwargs

	def _update_params(self, key: str, value_to_generate):
		if key not in self.params:
			self.params[key] = value_to_generate()
			
		return self.params[key]

	def __call__(self) -> ArgsExtractor:
		if self.extraction_mode not in EXTRACTOR_BY_NAME:
			raise NotImplementedError()

		if self.extraction_mode in CONSTRAINTS_EXTRACTORS:
			self._update_params("dependency_parser", lambda: DependencyParserFactory(**self.params)())

		elif self.extraction_mode in VERB_REFERENCES_EXTRACTORS:
			dependency_parser = self._update_params("dependency_parser", lambda: DependencyParserFactory(**self.params)())
			verb_translator = self._update_params("verb_translator", lambda: VerbTranslatorFactory(**self.params)())
			arg_encoder = self._update_params("arg_encoder", lambda: ArgumentEncoderFactory(**self.params)())
			method_params = self._update_params(
				"method_params", lambda: EXTRACTORS_CONFIG.METHOD_PARAMS_BY_EXTRACTOR[self.extraction_mode])

			self._update_params("references_by_verb", lambda: get_references_by_predicate(
				extractions=load_extraction_references(
					path=EXTRACTORS_CONFIG.REFERENCES_PATH_BY_ENCODER[arg_encoder_to_tuple_id(arg_encoder)],
					extracted_file_handler=ExtractedFileHandler(dependency_parser),
					consider_pp_type=method_params.consider_pp_type
				),
				verb_translator=verb_translator,
				normalize=method_params.already_normalized
			))

		return EXTRACTOR_BY_NAME[self.extraction_mode](**self.params)

	@staticmethod
	def expand_parser(arg_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
		arg_parser.add_argument(
			"--extraction-mode", choices=list(EXTRACTOR_BY_NAME.keys()), default=EXTRACTORS_CONFIG.EXTRACTOR,
			help="Defines the method of arguments extraction"
		)
		arg_parser.add_argument(
			"--ignore-nomlex-cache", "-c", action="store_true",
			help="Ignores cached nomlex adaptation files instead of recreating them"
		)
		arg_parser.add_argument(
			"--nomlex-version", type=NomlexVersion, default=EXTRACTORS_CONFIG.NOMLEX_VERSION,
			help="NOMLEX's lexicon version"
		)
		arg_parser = DependencyParserFactory.expand_parser(arg_parser)
		arg_parser = VerbTranslatorFactory.expand_parser(arg_parser)
		arg_parser = ArgumentEncoderFactory.expand_parser(arg_parser)
		return arg_parser
