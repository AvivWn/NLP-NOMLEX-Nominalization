import numpy as np

from yet_another_verb.arguments_extractor.extractors.verb_references_based.similarity_scorer.similarity_scorer import \
	SimilarityScorer
from yet_another_verb.arguments_extractor.extractors.verb_references_based.similarity_scorer.utils import normalize_vec, \
	dot_product


class CosineScorer(SimilarityScorer):
	def __init__(self, normalized_left=False, normalized_right=False):
		self.normalized_left = normalized_left
		self.normalized_right = normalized_right

	def score(self, vector1: np.array, vector2: np.array) -> np.ndarray:
		if not self.normalized_left:
			vector1 = normalize_vec(vector1)
		if not self.normalized_right:
			vector2 = normalize_vec(vector2)

		return dot_product(vector1, vector2)
