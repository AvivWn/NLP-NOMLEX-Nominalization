from .builders.extractor_builder import ExtractorBuilder
from .builders.nomlex_noun_extractor_builder import NomlexNounExtractorBuilder
from .builders.nomlex_verb_extractor_builder import NomlexVerbExtractorBuilder
from .candidates_finders.candidates_finder import CandidatesFinder
from .structural_matchers.structural_matcher import StructuralMatcher
from .extractions_filters.extractions_filter import ExtractionsFilter
from .extraction.predicate import Predicate
from noun_as_verb.verb_noun_translator.verb_noun_translator import VerbNounTranslator
from noun_as_verb.utils import get_dependency_tree
from .word_level_args_extractor import WordLevelArgsExtractor
from noun_as_verb.utils import is_noun, is_verb


class ArgumentsExtractor:
	verb_noun_transltor: VerbNounTranslator
	candidates_finder: CandidatesFinder
	structural_matcher: StructuralMatcher
	extractions_filter: ExtractionsFilter

	def __init__(self, verb_extractor_builder: ExtractorBuilder, noun_extractor_builder: ExtractorBuilder):
		self.noun_args_extractor = WordLevelArgsExtractor(verb_extractor_builder)
		self.verb_args_extractor = WordLevelArgsExtractor(noun_extractor_builder)

	@staticmethod
	def get_rule_extractor():
		return ArgumentsExtractor(NomlexVerbExtractorBuilder(), NomlexNounExtractorBuilder())

	def extract_argumnents(self, sentence, word_condition):
		doc = get_dependency_tree(sentence)
		extractions_per_word = {}
		none_predicates = []

		for token in doc:
			if not word_condition(token):
				continue

			if is_noun(token):
				word_extractor = self.noun_args_extractor
			elif is_verb(token):
				word_extractor = self.verb_args_extractor
			else:
				continue

			if word_extractor.is_predicate(token):
				predicate = Predicate(token)
				extractions_per_word[predicate] = word_extractor.extract_arguments(predicate)
			else:
				none_predicates.append(token)

		return extractions_per_word, none_predicates
