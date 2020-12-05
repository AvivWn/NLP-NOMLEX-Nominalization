from .predicate_detector import PredicateDetector


class NomlexPredicateDetector(PredicateDetector):
	def __init__(self, lexicon):
		super().__init__()
		self.lexicon = lexicon

	def is_predicate(self, token):
		return self.lexicon.is_contain(token)
