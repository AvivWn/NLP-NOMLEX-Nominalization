class ExtractionsFilter:
	def __init__(self):
		pass

	@staticmethod
	def _choose_longest(extractions):
		extractions.sort(key=lambda e: len(e.get_complements()), reverse=True)
		extractions = list(filter(lambda e: len(e) == len(extractions[0]), extractions))
		return extractions

	@staticmethod
	def _choose_informative(extractions):
		informative_extractions = []

		for e in extractions:
			found_more_general = False

			for other_e in extractions:
				if e.is_more_informative(other_e):
					found_more_general = True
					break

			if not found_more_general:
				informative_extractions.append(e)

		return informative_extractions

	@staticmethod
	def _uniqify(extractions):
		unique_extractions = []

		for e in extractions:
			if e in unique_extractions:
				continue

			unique_extractions.append(e)

		return unique_extractions

	def filter(self, extractions: list, predicate):
		longest_extractions = self._choose_longest(extractions)
		informative_extractions = self._choose_informative(longest_extractions)
		return self._uniqify(informative_extractions)
