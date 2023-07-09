from collections import defaultdict
from itertools import chain

from yet_another_verb.arguments_extractor.extraction.argument.extracted_argument import ExtractedArguments
from yet_another_verb.arguments_extractor.extractors.verb_references_based.argument_labelers.hypothesizer_args_labeler import \
	HypothesizerArgumentsLabeler
from yet_another_verb.arguments_extractor.extractors.verb_references_based.argument_labelers.utils import \
	TypesDistributions, get_k_largests_indices
from yet_another_verb.arguments_extractor.extractors.verb_references_based.method_params import MethodParams
from yet_another_verb.arguments_extractor.extractors.verb_references_based.verb_references.references_corpus import ReferencesCorpus
from yet_another_verb.arguments_extractor.extractors.verb_references_based.similarity_scorer.utils import \
	SimilaritiesByArg


class ArgumentsLabelerByArgumentKNN(HypothesizerArgumentsLabeler):
	def _hypothesize_types(
			self, args: ExtractedArguments, references_corpus: ReferencesCorpus,
			ref_similarities_by_arg: SimilaritiesByArg, method_params: MethodParams) -> TypesDistributions:
		type_distributions = []

		for arg in args:
			reference_types = list(
				chain(*[[arg_type] * len(scores) for arg_type, scores in ref_similarities_by_arg[arg].items()]))
			reference_similarities = list(chain(*ref_similarities_by_arg[arg].values()))
			closeset_reference_indices = get_k_largests_indices(reference_similarities, method_params.k_neighbors)

			type_distribution = defaultdict(list)
			for reference_index in closeset_reference_indices:
				reference_type = reference_types[reference_index]
				type_distribution[reference_type].append(reference_similarities[reference_index])

			type_distribution = {arg_type: sum(similarities) for arg_type, similarities in type_distribution.items()}
			type_distributions.append(type_distribution)

		return type_distributions
