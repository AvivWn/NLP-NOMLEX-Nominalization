import abc

from yet_another_verb.arguments_extractor.extraction.argument.extracted_argument import ExtractedArguments
from yet_another_verb.arguments_extractor.extractors.verb_references_based.method_params import MethodParams
from yet_another_verb.arguments_extractor.extractors.verb_references_based.verb_references.references_corpus import ReferencesCorpus
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText


class ArgumentsLabeler(abc.ABC):
	@abc.abstractmethod
	def label_arguments(
			self, args: ExtractedArguments, words: ParsedText,
			references_corpus: ReferencesCorpus, method_params: MethodParams):
		pass
