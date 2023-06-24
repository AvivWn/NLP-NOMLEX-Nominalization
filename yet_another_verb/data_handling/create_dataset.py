from os import makedirs
from argparse import ArgumentParser, Namespace
from os.path import dirname

from yet_another_verb.configuration.verbose_config import VERBOSE_CONFIG
from yet_another_verb.data_handling.creators_configs import DATASET_CREATOR_TYPE_BY_NAME, \
	DATASET_CREATOR_BY_TYPE, LIMITED_VERBS_BY_TYPE
from yet_another_verb.factories.encoder_factory import EncoderFactory
from yet_another_verb.factories.extractor_factory import ExtractorFactory
from yet_another_verb.factories.verb_translator_factory import VerbTranslatorFactory
from yet_another_verb.utils.print_utils import print_if_verbose, print_as_title_if_verbose
from yet_another_verb.utils.debug_utils import timeit


def show_args(args: Namespace):
	print_as_title_if_verbose("Parameters")
	for arg, value in vars(args).items():
		print_if_verbose(f"{arg}: {value}")


def create_dataset(args: Namespace):
	show_args(args)

	creator_type = DATASET_CREATOR_TYPE_BY_NAME[args.dataset_type]
	creator_args = vars(args)

	out_dataset_path = args.out_dataset_path
	makedirs(dirname(out_dataset_path), exist_ok=True)
	print_if_verbose(f"{creator_type.__name__}: {args.in_dataset_path} -> {out_dataset_path}")

	dataset_creator = DATASET_CREATOR_BY_TYPE.get(creator_type, lambda x: creator_type(**x))(creator_args)

	try:
		if args.overwrite or not dataset_creator.is_dataset_exist(out_dataset_path):
			timeit(dataset_creator.create_dataset)(out_dataset_path)
			print_if_verbose("Generated Successfully")
		elif args.append:
			timeit(dataset_creator.append_dataset)(out_dataset_path)
			print_if_verbose("Appended Successfully")
		else:
			print_if_verbose("Skipped")
	except NotImplementedError:
		print_if_verbose("Skipped")


def main():
	arg_parser = ArgumentParser()
	arg_parser.add_argument("--dataset-type", choices=DATASET_CREATOR_TYPE_BY_NAME.keys(), required=True)
	arg_parser.add_argument("--overwrite", "-w", action="store_true")
	arg_parser.add_argument("--append", "-a", action="store_true")
	arg_parser.add_argument("--dataset-size", "-s", type=int, default=None)
	arg_parser.add_argument("--in-dataset-path", "-i", type=str, default="")
	arg_parser.add_argument("--out-dataset-path", "-o", type=str, required=True)
	arg_parser.add_argument("--verbose", "-v", action="store_true")

	# Encoded-extractions params
	arg_parser.add_argument("--limited-postags", type=str, nargs='+', default=[])
	arg_parser.add_argument("--limited-verbs-type", choices=LIMITED_VERBS_BY_TYPE.keys(), default=None)

	ExtractorFactory.expand_parser(arg_parser)
	VerbTranslatorFactory.expand_parser(arg_parser)
	EncoderFactory.expand_parser(arg_parser)

	args, _ = arg_parser.parse_known_args()
	VERBOSE_CONFIG.VERBOSE = args.verbose
	create_dataset(args)


if __name__ == "__main__":
	main()
