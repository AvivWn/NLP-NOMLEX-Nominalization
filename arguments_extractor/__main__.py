import sys

from arguments_extractor.lisp_to_json.lisp_to_json import lisp_to_json
from arguments_extractor.model_based.dataset_creator import DatasetCreator
from arguments_extractor.model_based.types_predictor import TypesPredictor
from arguments_extractor.arguments_extractor import ArgumentsExtractor
from arguments_extractor.test import test
from arguments_extractor.utils import separate_line_print, timeit
from arguments_extractor import config

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

	extractor_function = timeit(extractor_func)
	extractions_per_verb, extractions_per_nom = extractor_function(test_extractor, sentence)

	print("--------------------------------\nVERB:")
	separate_line_print(extractions_per_verb)

	print("--------------------------------\nNOM:")
	separate_line_print(extractions_per_nom)

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
	# export LC_CTYPE=en_US.UTF-8
	# export LC_ALL=en_US.UTF-8

	config.LOAD_DATASET = True
	config.LOAD_LEXICON = True
	arguments_extractor = ArgumentsExtractor(config.LEXICON_FILE_NAME)
	dataset_creator = DatasetCreator(arguments_extractor)

	if "-example" in sys.argv:
		dataset_creator.create_examples_dataset(config.WIKI_SENTENCES_PATH, config.EXAMPLE_SENTENCES_PATH)

	elif "-args" in sys.argv:
		# ls -d arguments_extractor/model_based/data/shuffled_wiki_files/*.parsed | parallel --jobs 7 --u "python -m arguments_extractor -datasets -args"
		dataset_creator.create_args_dataset(sys.argv[-1])

	elif "-nouns" in sys.argv:
		# ls -d arguments_extractor/model_based/data/shuffled_wiki_files/*.parsed | parallel --jobs 13 --u "python -m arguments_extractor -datasets -nouns"
		dataset_creator.create_nouns_dataset(sys.argv[-1])

	elif "-parse" in sys.argv:
		# ls -d arguments_extractor/model_based/data/shuffled_wiki_files/*.txt | parallel --jobs 13 --u "python -m arguments_extractor -datasets -parse"
		dataset_creator.create_parsed_dataset(sys.argv[-1])

	# Use this to merge all the created datasets
	# cat shuffled_wiki_files/*_args.csv > shuffled_wiki_sentences_args.csv
	# cat shuffled_wiki_files/*_nouns.csv > shuffled_wiki_sentences_nouns.csv



if "-train" in sys.argv:
	arguments_predictor = TypesPredictor()
	arguments_predictor.train()