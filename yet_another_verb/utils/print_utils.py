def print_extracion(extraction_repr, indentation=""):
	if not extraction_repr:
		print(indentation + "-")
		return

	if isinstance(extraction_repr, list):
		for e in extraction_repr:
			print_extracion(e, indentation)
	else:
		print(indentation + str(extraction_repr))


def print_multi_word_extraction(multi_word_extraction_repr):
	indentation = " " * 4

	for word, extraction_repr in multi_word_extraction_repr.items():
		print(word + ":")
		print_extracion(extraction_repr, indentation)
