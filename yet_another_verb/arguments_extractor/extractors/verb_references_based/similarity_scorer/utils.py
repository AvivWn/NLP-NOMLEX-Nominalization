from typing import Dict

import numpy as np

from yet_another_verb.arguments_extractor.extraction import ArgumentType, ExtractedArgument
from yet_another_verb.arguments_extractor.extractors.verb_references_based.verb_references.references_corpus import ReferencesCorpus
from yet_another_verb.arguments_extractor.extractors.verb_references_based.similarity_scorer.similarity_scorer import \
    SimilarityScorer

Similarities = Dict[ArgumentType, np.array]
SimilaritiesByArg = Dict[ExtractedArgument, Similarities]


def normalize_vec(vector: np.array) -> np.array:
    norm = np.linalg.norm(vector)
    return np.divide(vector, norm, out=np.zeros_like(vector), where=norm != 0)


def dot_product(vector1: np.array, vector2: np.array) -> np.ndarray:
    return np.dot(vector1, vector2)


def calculate_similarities(arg, references_corpus: ReferencesCorpus, similarity_scorer: SimilarityScorer) \
        -> Similarities:
    vectors_by_type = references_corpus.encodings_by_arg_type
    return {arg_type: similarity_scorer.score(vectors, arg.encoding) for arg_type, vectors in vectors_by_type.items()}


def calculate_similarities_per_arg(args, references_corpus: ReferencesCorpus, similarity_scorer: SimilarityScorer) \
        -> SimilaritiesByArg:
    return {arg: calculate_similarities(arg, references_corpus, similarity_scorer) for arg in args}
