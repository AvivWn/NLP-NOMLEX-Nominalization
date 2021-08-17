def print_extraction(extraction_repr):
	for predicate, extractions in extraction_repr.items():
		if not extractions:
			continue

		print(predicate + ":")

		for e in extractions:
			print(" " * 4 + str(e))
