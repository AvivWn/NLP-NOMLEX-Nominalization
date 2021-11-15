def print_extraction(extraction_repr):
	indentation = " " * 4

	for predicate, extractions in extraction_repr.items():
		print(predicate + ":")

		if not extractions:
			print(indentation + "-")
			continue

		for e in extractions:
			print(indentation + str(e))
