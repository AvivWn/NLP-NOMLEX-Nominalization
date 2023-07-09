import numpy as np

from yet_another_verb.arguments_extractor.extraction.argument.extracted_argument import ExtractedArguments
from yet_another_verb.arguments_extractor.extractors.verb_references_based.argument_labelers.hypothesizer_args_labeler import \
	HypothesizerArgumentsLabeler
from yet_another_verb.arguments_extractor.extractors.verb_references_based.argument_labelers.utils import \
	TypesDistributions
from yet_another_verb.arguments_extractor.extractors.verb_references_based.method_params import MethodParams
from yet_another_verb.arguments_extractor.extractors.verb_references_based.verb_references.references_corpus import ReferencesCorpus
from yet_another_verb.arguments_extractor.extractors.verb_references_based.similarity_scorer.utils import \
	SimilaritiesByArg


class ArgumentsLabelerByAVG(HypothesizerArgumentsLabeler):
	def _hypothesize_types(
			self, args: ExtractedArguments, references_corpus: ReferencesCorpus,
			ref_similarities_by_arg: SimilaritiesByArg, method_params: MethodParams) -> TypesDistributions:
		ref_arg_vectors_by_type = references_corpus.encodings_by_arg_type  # padded_encodings_by_arg_type
		ref_avg_by_type = {
			arg_type: np.mean(ref_vectors, axis=0) for arg_type, ref_vectors in ref_arg_vectors_by_type.items()}

		score = method_params.similarity_scorer.score

		type_distributions = []
		for arg in args:
			type_distribution = {
				arg_type: score(arg.encoding, ref_avg_vector) for arg_type, ref_avg_vector in ref_avg_by_type.items()}
			type_distributions.append(type_distribution)

		return type_distributions
