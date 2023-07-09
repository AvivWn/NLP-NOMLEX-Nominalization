from yet_another_verb.arguments_extractor.extraction import ArgumentType
from yet_another_verb.arguments_extractor.extraction.argument.extracted_argument import ExtractedArguments
from yet_another_verb.arguments_extractor.extractors.verb_references_based.argument_labelers.args_labeler import \
	ArgumentsLabeler
from yet_another_verb.arguments_extractor.extractors.verb_references_based.method_params import MethodParams
from yet_another_verb.arguments_extractor.extractors.verb_references_based.verb_references.references_corpus import ReferencesCorpus
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText


class SingleTypeArgumentsLabeler(ArgumentsLabeler):
	def __init__(self, single_arg_type: ArgumentType):
		self.single_arg_type = single_arg_type

	def label_arguments(
			self, args: ExtractedArguments, words: ParsedText,
			references_corpus: ReferencesCorpus, method_params: MethodParams):
		for arg in args:
			arg.arg_type = self.single_arg_type
