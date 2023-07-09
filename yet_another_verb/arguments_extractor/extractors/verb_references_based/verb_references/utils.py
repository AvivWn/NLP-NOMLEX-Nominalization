from collections import defaultdict, namedtuple
from itertools import chain
from os.path import exists
from typing import Optional, List

import numpy as np

from yet_another_verb.arguments_extractor.extraction import Extractions, ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.utils.extraction_utils import aggregate_by_predicate, \
	get_arg_types_in_extractions, get_extractions_with_modified_pp, separate_extractions_with_repeated_arg_type, \
	reduce_extractions_by_arg_types, convert_encodings_to_numpy
from yet_another_verb.arguments_extractor.extraction.utils.reconstruction import reconstruct_extractions
from yet_another_verb.arguments_extractor.extractors.verb_references_based.argument_labelers.utils import \
	get_k_largests_indices
from yet_another_verb.arguments_extractor.extractors.verb_references_based.similarity_scorer.similarity_scorer import \
	SimilarityScorer
from yet_another_verb.arguments_extractor.extractors.verb_references_based.similarity_scorer.utils import normalize_vec, \
	calculate_similarities
from yet_another_verb.arguments_extractor.extractors.verb_references_based.verb_references.references_corpus import \
	ReferencesCorpus, ReferencesByPredicate
from yet_another_verb.configuration.extractors_config import EXTRACTORS_CONFIG
from yet_another_verb.data_handling import ExtractedFileHandler
from yet_another_verb.word_to_verb.verb_translator import VerbTranslator


ScoredReference = namedtuple("ScoredReference", ["extraction", "score"])


def agg_encodings_and_indices(encoded_exts, limited_arg_types, normalize=False):
	agg_encodings = defaultdict(list)
	agg_ext_indices = defaultdict(list)

	for ext_idx, encoded_ext in enumerate(encoded_exts):
		# repeated args check
		ext_arg_types = [arg.arg_type for arg in encoded_ext.all_args]
		assert len(ext_arg_types) == len(set(ext_arg_types)), "Founded repeated arg types in the given extrraction"

		founded_arg_types = []

		for arg in encoded_ext.all_args:
			founded_arg_types.append(arg.arg_type)
			if arg.arg_type in limited_arg_types:
				enc = normalize_vec(arg.encoding) if normalize else arg.encoding
				agg_encodings[arg.arg_type].append([enc])
				agg_ext_indices[arg.arg_type].append(ext_idx)

	return {arg_type: np.concatenate(agg_encodings[arg_type]) for arg_type in agg_encodings}, \
		{arg_type: np.array(agg_ext_indices[arg_type]) for arg_type in agg_ext_indices}


def get_references_by_predicate(
		extractions: Extractions, verb_translator: Optional[VerbTranslator] = None, normalize=True) -> ReferencesByPredicate:
	references_corpus_by_predicate = {}
	references_by_predicate = aggregate_by_predicate(extractions, verb_translator)

	for predicate, references in references_by_predicate.items():
		arg_types = get_arg_types_in_extractions(references)
		encodings_by_arg_type, ext_indices_by_arg_type = agg_encodings_and_indices(references, arg_types, normalize=normalize)
		references_corpus_by_predicate[predicate] = ReferencesCorpus(references, encodings_by_arg_type, ext_indices_by_arg_type)

	return references_corpus_by_predicate


def load_extraction_references(
		path: str, extracted_file_handler: ExtractedFileHandler, consider_pp_type: bool) -> Extractions:
	caching_path = f"{path}-references-pp_typed" if consider_pp_type else f"{path}-references"

	if exists(caching_path):
		return extracted_file_handler.load(caching_path)

	extractions = extracted_file_handler.load(path)
	convert_encodings_to_numpy(extractions)
	extractions = reconstruct_extractions(extractions)
	extractions = reduce_extractions_by_arg_types(
		extractions, arg_types=EXTRACTORS_CONFIG.REFERENCES_ARG_TYPES, reconstruct=False)
	extractions = get_extractions_with_modified_pp(extractions, consider_pp_type, reconstruct=False)
	extractions = separate_extractions_with_repeated_arg_type(extractions)
	extracted_file_handler.save(caching_path, extractions)

	return extractions


def get_closest_references(
		arg: ExtractedArgument, references_corpus: ReferencesCorpus, similarity_scorer: SimilarityScorer,
		k_closest=1, arg_types=None) -> List[ScoredReference]:
	ref_similarities_by_type = calculate_similarities(arg, references_corpus, similarity_scorer)

	reference_idxs = list(chain(*references_corpus.ext_indices_by_arg_type.values()))
	reference_types = list(chain(*[[arg_type] * len(scores) for arg_type, scores in ref_similarities_by_type.items()]))
	reference_similarities = list(chain(*ref_similarities_by_type.values()))

	for ref_index in range(len(reference_similarities)):
		if arg_types is not None and reference_types[ref_index] not in arg_types:
			reference_similarities[ref_index] = -np.inf

	closest_ref_indices = get_k_largests_indices(reference_similarities, k_closest)

	scored_references = []
	for i, ref_index in enumerate(closest_ref_indices):
		ref_similarity = reference_similarities[ref_index]

		# ignore paddings
		if ref_similarity in [0.0, -np.inf]:
			continue

		ext_reference = references_corpus.extractions[reference_idxs[ref_index]]
		scored_references.append(ScoredReference(extraction=ext_reference, score=ref_similarity))

	return scored_references
