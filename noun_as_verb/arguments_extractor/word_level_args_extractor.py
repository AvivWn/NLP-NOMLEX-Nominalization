from .builders.extractor_builder import ExtractorBuilder
from .predicate_detectors.predicate_detector import PredicateDetector
from .candidates_finders.candidates_finder import CandidatesFinder
from .structural_matchers.structural_matcher import StructuralMatcher
from .extractions_filters.extractions_filter import ExtractionsFilter
from .extraction.predicate import Predicate


class WordLevelArgsExtractor:
	predicate_detector: PredicateDetector
	candidates_finder: CandidatesFinder
	structural_matcher: StructuralMatcher
	extractions_filter: ExtractionsFilter

	def __init__(self, extractor_builder: ExtractorBuilder):
		self.predicate_detector = extractor_builder.get_predicate_detector()
		self.candidates_finder = extractor_builder.get_candidates_finder()
		self.structural_matcher = extractor_builder.get_structural_matcher()
		self.extractions_filter = extractor_builder.get_extractions_filter()

	def is_predicate(self, token):
		return self.predicate_detector.is_predicate(token)

	def extract_arguments(self, predicate: Predicate):
		candidates = self.candidates_finder.find_candidates(predicate)
		extractions = self.structural_matcher.find_structures(candidates, predicate)
		extractions = self.extractions_filter.filter(extractions, predicate)
		return extractions
