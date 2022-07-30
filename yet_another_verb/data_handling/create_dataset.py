from os import makedirs
from os.path import join
from argparse import ArgumentParser, Namespace

from yet_another_verb.configuration.verbose_config import VERBOSE_CONFIG
from yet_another_verb.data_handling.creators_configs import DatasetConfig, DATASET_CONFIGS_BY_TYPE, \
	DATASET_CREATORS_BY_TYPE, EXTENSION_BY_CREATOR_TYPE, DIR_BY_CREATORE_TYPE
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.factories.extractor_factory import ExtractorFactory
from yet_another_verb.factories.verb_translator_factory import VerbTranslatorFactory
from yet_another_verb.utils.debug_utils import timeit
from yet_another_verb.utils.print_utils import print_if_verbose, print_as_title_if_verbose


def show_args(args: Namespace):
	print_as_title_if_verbose("Parameters")
	for arg, value in vars(args).items():
		print_if_verbose(f"{arg}: {value}")


def get_output_path(data_dir: str, basename: str, dataset_config: DatasetConfig) -> str:
	output_dir = join(data_dir, DIR_BY_CREATORE_TYPE.get(dataset_config.creator_type, ""), dataset_config.subdir_name)
	makedirs(output_dir, exist_ok=True)

	file_extension = EXTENSION_BY_CREATOR_TYPE[dataset_config.creator_type]
	return join(output_dir, f"{basename}.{file_extension}")


def create_dataset(args: Namespace):
	show_args(args)

	for dataset_config in DATASET_CONFIGS_BY_TYPE[args.dataset_type]:
		creator_type = dataset_config.creator_type
		creator_generator = DATASET_CREATORS_BY_TYPE.get(creator_type, lambda x: creator_type(**x))

		creator_args = vars(args)
		creator_args.update(dataset_config.args)
		dataset_creator = creator_generator(creator_args)

		out_path = get_output_path(args.data_dir, args.out_dataset_basename, dataset_config)
		print_if_verbose(f"{type(dataset_creator).__name__}: {args.in_dataset_path} -> {out_path}")

		try:
			if args.overwrite or not dataset_creator.is_dataset_exist(out_path):
				timeit(dataset_creator.create_dataset)(out_path)
				print_if_verbose("Generated Successfully")
			elif args.append:
				timeit(dataset_creator.append_dataset)(out_path)
				print_if_verbose("Appended Successfully")
			else:
				print_if_verbose("Skipped")
		except NotImplementedError:
			print_if_verbose("Skipped")

		args.in_dataset_path = out_path


def main():
	arg_parser = ArgumentParser()
	arg_parser.add_argument("--dataset-type", choices=DATASET_CONFIGS_BY_TYPE.keys(), required=True)
	arg_parser.add_argument("--overwrite", "-w", action="store_true")
	arg_parser.add_argument("--append", "-a", action="store_true")
	arg_parser.add_argument("--dataset-size", "-s", type=int, default=None)
	arg_parser.add_argument("--in-dataset-path", "-i", type=str, default="")
	arg_parser.add_argument("--out-dataset-basename", "-o", type=str, required=True)
	arg_parser.add_argument("--data-dir", "-d", type=str, default="")
	arg_parser.add_argument("--verbose", "-v", action="store_true")

	DependencyParserFactory.expand_parser(arg_parser)
	ExtractorFactory.expand_parser(arg_parser)
	VerbTranslatorFactory.expand_parser(arg_parser)

	args, _ = arg_parser.parse_known_args()
	VERBOSE_CONFIG.VERBOSE = args.verbose
	create_dataset(args)


if __name__ == "__main__":
	main()
