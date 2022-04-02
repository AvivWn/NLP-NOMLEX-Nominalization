from argparse import ArgumentParser, Namespace

from yet_another_verb.configuration.verbose_config import VERBOSE_CONFIG
from yet_another_verb.factories.verb_translator_factory import VerbTranslatorFactory
from yet_another_verb.ml.datasets_creator.dataset_creator import DatasetCreator
from yet_another_verb.ml.datasets_creator import WikiDatasetCreator, ParsedDatasetCreator, ExtractedDatasetCreator, \
	BIOArgsDatasetCreator
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.factories.extractor_factory import ExtractorFactory
from yet_another_verb.nomlex.constants.argument_type import PP_ARG_TYPES, NP_ARG_TYPES
from yet_another_verb.nomlex.constants.word_postag import VERB_POSTAGS, NOUN_POSTAGS
from yet_another_verb.utils.debug_utils import timeit

DATASET_TO_CREATOR = {
	"wiki40b": lambda args: WikiDatasetCreator(**vars(args)),
	"ud-parsed": lambda args: ParsedDatasetCreator(
		**vars(args),
		dependency_parser=DependencyParserFactory(**vars(args))()),
	"extracted": lambda args: ExtractedDatasetCreator(
		**vars(args),
		args_extractor=ExtractorFactory(**vars(args))(),
		dependency_parser=DependencyParserFactory(**vars(args))()),
	"bio-args": lambda args: BIOArgsDatasetCreator(
		**vars(args),
		limited_postags=VERB_POSTAGS,
		limited_types=NP_ARG_TYPES + PP_ARG_TYPES,
		use_base_verb=True,
		avoid_outside_tag=True,
		verb_translator=VerbTranslatorFactory(**vars(args))()),
}


def create_dataset(args: Namespace):
	out_path = args.out_dataset_path

	dataset_creator: DatasetCreator = DATASET_TO_CREATOR[args.dataset_type](args)

	if not args.overwrite and dataset_creator.is_dataset_exist(out_path):
		return

	timeit(dataset_creator.create_dataset)(out_path)


def main():
	arg_parser = ArgumentParser()
	arg_parser.add_argument("--dataset-type", choices=DATASET_TO_CREATOR.keys(), default="extracted")
	arg_parser.add_argument("--overwrite", "-w", action="store_true")
	arg_parser.add_argument("--dataset-size", "-s", type=int, default=None)
	arg_parser.add_argument("--out-dataset-path", "-o", type=str, required=True)
	arg_parser.add_argument("--in-dataset-path", "-i", type=str, default="")
	arg_parser.add_argument("--verbose", "-v", action="store_true")

	DependencyParserFactory.expand_parser(arg_parser)
	ExtractorFactory.expand_parser(arg_parser)
	VerbTranslatorFactory.expand_parser(arg_parser)

	args, _ = arg_parser.parse_known_args()
	VERBOSE_CONFIG.VERBOSE = args.verbose
	create_dataset(args)


if __name__ == "__main__":
	main()
