from argparse import ArgumentParser, Namespace

from yet_another_verb.arguments_extractor.extractors.nomlex_args_extractor import NomlexArgsExtractor
from yet_another_verb.arguments_extractor.extraction.representation.parsed_str_representation import \
	ParsedStrRepresentation
from yet_another_verb.nomlex.nomlex_version import NomlexVersion
from yet_another_verb.utils.debug_utils import timeit
from yet_another_verb.utils.print_utils import print_extraction
from yet_another_verb.configuration.nomlex_config import NOMLEX_CONFIG


def extract(args: Namespace):
	NOMLEX_CONFIG.USE_CACHE = not args.ignore_cache
	NOMLEX_CONFIG.NOMLEX_VERSION = args.nomlex_version

	if args.extraction_mode == "nomlex":
		args_extractor = NomlexArgsExtractor()
	else:
		return

	limited_idxs = None if args.word_idx is None else [args.word_idx]

	parsed_text = args_extractor.preprocess(args.text)
	extractions_per_idx = timeit(args_extractor.extract_multiword)(parsed_text, limited_idxs)
	str_repr = ParsedStrRepresentation(parsed_text).represent_dict(extractions_per_idx)
	print_extraction(str_repr)


def main():
	arg_parser = ArgumentParser()
	arg_parser.add_argument(
		"--text", "-t", type=str, default="", required=True,
		help="The text used to extract arguments"
	)
	arg_parser.add_argument(
		"--extraction-mode", "-m", choices=["nomlex"], default="nomlex",
		help="Defines the method of arguments extraction"
	)
	arg_parser.add_argument(
		"--word-idx", "-i", type=int, default=None,
		help="Specific index of a word to focus on"
	)
	arg_parser.add_argument(
		"--ignore-cache", "-c", action="store_true",
		help="Use cached files instead of recreating them"
	)
	arg_parser.add_argument(
		"--nomlex-version", "-v", type=NomlexVersion, default=NomlexVersion.V2,
		help="NOMLEX's lexicon version"
	)
	args, _ = arg_parser.parse_known_args()
	extract(args)


if __name__ == "__main__":
	main()
