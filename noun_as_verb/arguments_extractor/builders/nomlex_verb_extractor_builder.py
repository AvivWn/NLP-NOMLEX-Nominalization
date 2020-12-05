from .extractor_builder import ExtractorBuilder
from ..predicate_detectors.nomlex_predicate_detector import NomlexPredicateDetector
from ..candidates_finders.candidates_finder import CandidatesFinder
from ..structural_matchers.nomlex_matcher import NomlexMatcher
from ..extractions_filters.extractions_filter import ExtractionsFilter
from noun_as_verb.lexicon_representation.lexicon import Lexicon
from noun_as_verb import config


class NomlexVerbExtractorBuilder(ExtractorBuilder):
	def __init__(self):
		super().__init__()
		self.lexicon = Lexicon(config.LEXICON_FILE_NAME, is_verb=True)

	def get_predicate_detector(self):
		return NomlexPredicateDetector(self.lexicon)

	def get_candidates_finder(self):
		return CandidatesFinder()

	def get_structural_matcher(self):
		return NomlexMatcher(self.lexicon)

	def get_extractions_filter(self):
		return ExtractionsFilter()
