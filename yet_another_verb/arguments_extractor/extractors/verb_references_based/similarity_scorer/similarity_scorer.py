import abc


class SimilarityScorer(abc.ABC):
	def score(self, vector1, vector2):
		pass
