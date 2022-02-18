from argparse import ArgumentParser, Namespace

from yet_another_verb.ml.datasets_creator.dataset_creator import DatasetCreator
from yet_another_verb.ml.datasets_creator import WikiDatasetCreator, ParsedDatasetCreator, ExtractedDatasetCreator
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.factories.extractor_factory import ExtractorFactory


DATASET_TO_CREATOR = {
	"wiki40b": lambda args: WikiDatasetCreator(args.dataset_size),
	"ud-parsed": lambda args: ParsedDatasetCreator(
		args.in_path, DependencyParserFactory(**vars(args))(), args.dataset_size),
	"extracted": lambda args: ExtractedDatasetCreator(
		args.in_path,
		ExtractorFactory(**vars(args))(), DependencyParserFactory(**vars(args))(),
		args.dataset_size
	)
}


def create_dataset(args: Namespace):
	print(args.in_path, args.out_path)
	out_path = args.out_path

	dataset_creator: DatasetCreator = DATASET_TO_CREATOR[args.dataset_type](args)

	if not args.overwrite and dataset_creator.is_dataset_exist(out_path):
		return

	dataset_creator.create_dataset(out_path)


def main():
	arg_parser = ArgumentParser()
	arg_parser.add_argument("--dataset-type", choices=DATASET_TO_CREATOR.keys(), default="extracted")
	arg_parser.add_argument("--overwrite", "-w", action="store_true")
	arg_parser.add_argument("--dataset-size", "-s", type=int, default=None)
	arg_parser.add_argument("--out-path", "-o", type=str, required=True)
	arg_parser.add_argument("--in-path", "-i", type=str, default="")

	DependencyParserFactory.expand_parser(arg_parser)
	ExtractorFactory.expand_parser(arg_parser)

	args, _ = arg_parser.parse_known_args()
	create_dataset(args)


if __name__ == "__main__":
	main()
