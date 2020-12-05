import sys

from noun_as_verb.lisp_to_json import lisp_to_json
from noun_as_verb.model_based.dataset_creator import DatasetCreator
from noun_as_verb.model_based.types_predictor import TypesPredictor
from noun_as_verb.arguments_extractor.sentence_level_args_extractor import ArgumentsExtractor
from noun_as_verb.test import test
from noun_as_verb.utils import separate_line_print, timeit, get_dependency_tree
from noun_as_verb import config

def main():
	# Generation of lexicons and datasets can be forced
	if "-f" in sys.argv:
		config.LOAD_DATASET = False
		config.LOAD_LEXICON = False
		config.REWRITE_TEST = True
		config.IGNORE_PROCESSED_DATASET = False

	# DEBUG mode
	if "-debug" in sys.argv:
		config.DEBUG = True



	if "-lispToJson" in sys.argv:
		if not config.LOAD_LEXICON:
			ArgumentsExtractor(config.LEXICON_FILE_NAME)
		else:
			lisp_to_json(config.LEXICON_FILE_NAME)



	if "-rule" in sys.argv:
		extractor_func = ArgumentsExtractor.rule_based_extraction
	elif "-model" in sys.argv:
		extractor_func = ArgumentsExtractor.model_based_extraction
	elif "-hybrid" in sys.argv:
		extractor_func = ArgumentsExtractor.hybrid_based_extraction
	else: # default is rule-based
		extractor_func = ArgumentsExtractor.rule_based_extraction

	if "-extract" in sys.argv:
		sentence = sys.argv[-1]
		test_extractor = ArgumentsExtractor(config.LEXICON_FILE_NAME)

		dependency_tree = get_dependency_tree(sentence)
		extractor_function = timeit(extractor_func)
		extractions_per_verb, extractions_per_nom = extractor_function(test_extractor, dependency_tree)

		print("--------------------------------\nVERB:")
		separate_line_print(extractions_per_verb)

		print("--------------------------------\nNOM:")
		separate_line_print(extractions_per_nom)

		arguments_extractor = ArgumentsExtractor(config.LEXICON_FILE_NAME)
		dataset_creator = DatasetCreator(arguments_extractor)
		x = dataset_creator.get_nouns_samples(dependency_tree, {l:0 for l in {"SUBJECT", "OBJECT", "NONE", "NOT-NOM", "VERB-NOM", "PP", "IND-OBJ"}}, None)
		print(x)

	if "-search" in sys.argv:
		sentence = sys.argv[-1]
		test_extractor = ArgumentsExtractor(config.LEXICON_FILE_NAME)

		sentences_file = open(config.EXAMPLE_SENTENCES_PATH)
		searched_args = test_extractor.get_searched_args(sentence, extractor_func)

		search_function = timeit(test_extractor.search_matching_extractions)
		matching_extractions = search_function(searched_args, sentences_file, extractor_func, limited_results=5)
		separate_line_print(matching_extractions)

	if "-test" in sys.argv:
		test_extractor = ArgumentsExtractor(config.LEXICON_FILE_NAME)
		test(test_extractor, extractor_func)



	if "-datasets" in sys.argv:
		config.LOAD_DATASET = True
		config.LOAD_LEXICON = True
		arguments_extractor = ArgumentsExtractor(config.LEXICON_FILE_NAME)
		dataset_creator = DatasetCreator(arguments_extractor)
		in_path = sys.argv[-1]

		if "-sentences" in sys.argv:
			dataset_creator.create_sentences_dataset(in_path)
		elif "-parse" in sys.argv:
			dataset_creator.create_parsed_dataset(in_path)
		elif "-example" in sys.argv:
			dataset_creator.dataset_size = 100000
			dataset_creator.create_examples_dataset(in_path)
		elif "-combined" in sys.argv:
			dataset_creator.dataset_size = 100000
			dataset_creator.create_combined_dataset(in_path)
		elif "-args" in sys.argv:
			dataset_creator.dataset_size = 100000
			dataset_creator.create_args_dataset(in_path)
		elif "-nouns" in sys.argv:
			dataset_creator.dataset_size = 100000
			dataset_creator.create_nouns_dataset(in_path)



	if "-train" in sys.argv:
		#arguments_predictor = TypesPredictor({"SUBJECT", "OBJECT", "NONE"})
		arguments_predictor = TypesPredictor({"SUBJECT", "OBJECT", "NONE", "NOT-NOM", "VERB-NOM", "PP", "IND-OBJ"})
		arguments_predictor.train()

if __name__ == "__main__":
	main()