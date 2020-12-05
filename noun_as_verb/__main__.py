from argparse import ArgumentParser

from .lisp_to_json.lisp_to_json import lisp_to_json
from .arguments_extractor.sentence_level_args_extractor import ArgumentsExtractor
from .arguments_extractor.extraction.representation.span_representation import SpanRepresentation
from .utils import separate_line_print, timeit, get_dependency_tree
from . import config


def main():
	arg_parser = ArgumentParser()
	arg_parser.add_argument("--mode", choices=["rule", "model", "hybrid", "transfer"], default="rule")
	arg_parser.add_argument("--action", choices=["extract", "lispToJson", "train"], default="extract")
	arg_parser.add_argument("--sentence", type=str, default="")
	arg_parser.add_argument("--force", action="store_true")
	arg_parser.add_argument("--debug", action="store_true")
	args = arg_parser.parse_args()

	# Generation of lexicons and datasets can be forced
	if args.force:
		config.LOAD_DATASET = False
		config.LOAD_LEXICON = False
		config.REWRITE_TEST = True
		config.IGNORE_PROCESSED_DATASET = False

	# DEBUG mode
	if args.debug:
		config.DEBUG = True

	extractor_builder_func = ArgumentsExtractor.get_rule_extractor

	if args.action == "lispToJson":
		lisp_to_json(config.LEXICON_FILE_NAME)

	elif args.action == "extract":
		dependency_tree = get_dependency_tree(args.sentence)

		args_extractor = extractor_builder_func()
		extractor_function = timeit(args_extractor.extract_arguments)
		predicate2extractions = extractor_function(dependency_tree)
		predicate2extractions = SpanRepresentation().represent_dict(predicate2extractions)
		separate_line_print(predicate2extractions)


if __name__ == "__main__":
	main()
