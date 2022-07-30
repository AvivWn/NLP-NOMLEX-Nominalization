from yet_another_verb.configuration.verbose_config import VERBOSE_CONFIG


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
		print(str(word) + ":")
		print_extracion(extraction_repr, indentation)


def print_if_verbose(*args):
	if VERBOSE_CONFIG.VERBOSE:
		print(*args)


def print_as_title_if_verbose(*args):
	print_if_verbose(f"{'-'*10}", *args, f"{'-'*10}")
