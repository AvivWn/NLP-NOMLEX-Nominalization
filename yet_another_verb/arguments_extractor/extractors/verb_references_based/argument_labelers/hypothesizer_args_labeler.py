from yet_another_verb.arguments_extractor.extraction.argument.extracted_argument import ExtractedArguments
from yet_another_verb.arguments_extractor.extractors.verb_references_based.argument_labelers.args_labeler import \
	ArgumentsLabeler
from yet_another_verb.arguments_extractor.extractors.verb_references_based.argument_labelers.utils import \
	TypesDistributions, label_args_by_distributions, label_redundant_by_threshold
from yet_another_verb.arguments_extractor.extractors.verb_references_based.method_params import MethodParams
from yet_another_verb.arguments_extractor.extractors.verb_references_based.verb_references.references_corpus import ReferencesCorpus
from yet_another_verb.arguments_extractor.extractors.verb_references_based.similarity_scorer.utils import \
	calculate_similarities_per_arg, SimilaritiesByArg
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText


class HypothesizerArgumentsLabeler(ArgumentsLabeler):
	def _hypothesize_types(
			self, args: ExtractedArguments, references_corpus: ReferencesCorpus,
			ref_similarities_by_arg: SimilaritiesByArg, method_params: MethodParams) -> TypesDistributions:
		raise NotImplementedError()

	def label_arguments(
			self, args: ExtractedArguments, words: ParsedText,
			references_corpus: ReferencesCorpus, method_params: MethodParams):
		ref_similarities_by_arg = calculate_similarities_per_arg(args, references_corpus, method_params.similarity_scorer)
		redundant_args = label_redundant_by_threshold(args, ref_similarities_by_arg, method_params.redundant_threshold)

		args = [arg for arg in args if arg not in redundant_args]

		if len(args) != 0:
			type_distributions = self._hypothesize_types(
				args,
				references_corpus=references_corpus,
				ref_similarities_by_arg=ref_similarities_by_arg,
				method_params=method_params)

			label_args_by_distributions(args, type_distributions, method_params.max_repeated_args)
