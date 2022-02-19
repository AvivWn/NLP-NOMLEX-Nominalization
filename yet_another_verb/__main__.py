from argparse import ArgumentParser, Namespace

from yet_another_verb.arguments_extractor.extraction.representation.parsed_str_representation import \
	ParsedStrRepresentation
from yet_another_verb.configuration.verbose_config import VERBOSE_CONFIG
from yet_another_verb.factories.extractor_factory import ExtractorFactory
from yet_another_verb.utils.debug_utils import timeit
from yet_another_verb.utils.print_utils import print_multi_word_extraction
from yet_another_verb.configuration.extractors_config import EXTRACTORS_CONFIG


def extract(args: Namespace):
	EXTRACTORS_CONFIG.USE_NOMLEX_CACHE = not args.ignore_nomlex_cache

	args_extractor = ExtractorFactory(**vars(args))()

	limited_idxs = None if args.word_idx is None else [args.word_idx]
	multi_word_extraction = timeit(args_extractor.extract_multiword)(args.text, limited_idxs=limited_idxs)
	str_repr = ParsedStrRepresentation().represent_dict(multi_word_extraction)
	print_multi_word_extraction(str_repr)


def main():
	arg_parser = ArgumentParser()
	arg_parser.add_argument(
		"--text", "-t", type=str, default="", required=True,
		help="The text used to extract arguments"
	)
	arg_parser.add_argument(
		"--word-idx", "-i", type=int, default=None,
		help="Specific index of a word to focus on"
	)
	arg_parser.add_argument(
		"--verbose", "-v", action="store_true"
	)

	arg_parser = ExtractorFactory.expand_parser(arg_parser)

	args, _ = arg_parser.parse_known_args()
	VERBOSE_CONFIG.VERBOSE = args.verbose
	extract(args)


if __name__ == "__main__":
	main()
