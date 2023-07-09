from collections import defaultdict

import numpy as np

from yet_another_verb.arguments_extractor.extraction.argument.extracted_argument import ExtractedArguments
from yet_another_verb.arguments_extractor.extractors.verb_references_based.argument_labelers.hypothesizer_args_labeler import \
	HypothesizerArgumentsLabeler
from yet_another_verb.arguments_extractor.extractors.verb_references_based.argument_labelers.utils import \
	TypesDistributions, get_k_largests_indices
from yet_another_verb.arguments_extractor.extractors.verb_references_based.method_params import MethodParams
from yet_another_verb.arguments_extractor.extractors.verb_references_based.verb_references.references_corpus import ReferencesCorpus
from yet_another_verb.arguments_extractor.extractors.verb_references_based.similarity_scorer.utils import \
	SimilaritiesByArg


class ArgumentsLabelerByExtractionKNN(HypothesizerArgumentsLabeler):
	def _hypothesize_types(
			self, args: ExtractedArguments, references_corpus: ReferencesCorpus,
			ref_similarities_by_arg: SimilaritiesByArg, method_params: MethodParams) -> TypesDistributions:
		# Keep track on the closest extractions
		closest_similarity_scores = np.ones(method_params.k_neighbors) * -np.inf
		closest_type_distributions = [None] * method_params.k_neighbors

		for ordered_types in references_corpus.get_ordered_arg_types_combos(len(args)):
			relevant_args, relevant_types = [], []
			reference_arg_indexed_similarities = []

			for arg_type, arg in zip(ordered_types, args):
				if arg_type in ref_similarities_by_arg[arg]:
					relevant_args.append(arg)
					relevant_types.append(arg_type)
					ext_indices = list(references_corpus.ext_indices_by_arg_type[arg_type])
					ref_similarities = list(ref_similarities_by_arg[arg][arg_type])
					reference_arg_indexed_similarities.append(dict(zip(ext_indices, ref_similarities)))

			common_reference_indices = set.intersection(*[set(x.keys()) for x in reference_arg_indexed_similarities])

			if len(common_reference_indices) == 0:
				continue

			reference_arg_similarities = np.array(
				[[x[i] for i in common_reference_indices] for x in reference_arg_indexed_similarities])

			reference_ext_similarities = np.sum(reference_arg_similarities, axis=0)
			closeset_exts_indices = get_k_largests_indices(reference_ext_similarities, method_params.k_neighbors)

			for ref_idx in closeset_exts_indices:
				most_unsimilar_idx = np.argmin(closest_similarity_scores)

				if reference_ext_similarities[ref_idx] > closest_similarity_scores[most_unsimilar_idx]:
					closest_similarity_scores[most_unsimilar_idx] = reference_ext_similarities[ref_idx]
					closest_type_distributions[most_unsimilar_idx] = zip(relevant_args, relevant_types, [x[ref_idx] for x in reference_arg_similarities])

		type_distribution_by_arg = {}
		for type_distribution in closest_type_distributions:
			if type_distribution is None:
				continue

			for arg, arg_type, similarity in type_distribution:
				if arg not in type_distribution_by_arg:
					type_distribution_by_arg[arg] = defaultdict(list)

				type_distribution_by_arg[arg][arg_type].append(similarity)

		for arg, type_distribution in type_distribution_by_arg.items():
			type_distribution = {arg_type: sum(similarities) for arg_type, similarities in type_distribution.items()}
			type_distribution_by_arg[arg] = type_distribution

		return [type_distribution_by_arg.get(arg, {}) for arg in args]
