from .candidate import Candidate
from .ud_translator import LINK_TO_POS
from ..extraction.predicate import Predicate


class CandidatesFinder:
	@staticmethod
	def find_candidates(predicate: Predicate):
		referenced_token = predicate.get_token()

		candidates = []
		for sub_token in referenced_token.children:
			if sub_token.dep_ in LINK_TO_POS.keys():
				candidates.append(Candidate(sub_token, referenced_token))

		if predicate.is_noun():
			candidates.append(Candidate(referenced_token, referenced_token))

		return candidates
