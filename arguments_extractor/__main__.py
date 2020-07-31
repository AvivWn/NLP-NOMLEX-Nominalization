import sys

from arguments_extractor.lisp_to_json.lisp_to_json import lisp_to_json
from arguments_extractor.model_based.create_datasets import create_args_dataset, create_parsed_dataset, create_examples_dataset
from arguments_extractor.model_based.arguments_predictor import ArgumentsPredictor
from arguments_extractor.rule_based.lexicon import Lexicon
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
	extractor_function = ArgumentsExtractor.rule_based_extraction
elif "-model" in sys.argv:
	extractor_function = ArgumentsExtractor.model_based_extraction
elif "-hybrid" in sys.argv:
	extractor_function = ArgumentsExtractor.hybrid_based_extraction
else: # default is rule-based
	extractor_function = ArgumentsExtractor.rule_based_extraction

if "-extract" in sys.argv:
	sentence = sys.argv[-1]
	test_extractor = ArgumentsExtractor(config.LEXICON_FILE_NAME)

	extractor_function = timeit(extractor_function)
	extractions_per_verb, extractions_per_nom = extractor_function(test_extractor, sentence)

	print("--------------------------------\nVERB:")
	separate_line_print(extractions_per_verb)

	print("--------------------------------\nNOM:")
	separate_line_print(extractions_per_nom)

if "-test" in sys.argv:
	test_extractor = ArgumentsExtractor(config.LEXICON_FILE_NAME)
	test(test_extractor, extractor_function)



if "-datasets" in sys.argv:
	arguments_extractor = ArgumentsExtractor(config.LEXICON_FILE_NAME)

	if "-example" in sys.argv:
		create_examples_dataset(config.WIKI_SENTENCES_PATH, config.EXAMPLE_SENTENCES_PATH, arguments_extractor)

	elif "-args" in sys.argv:
		# Creating the training and testing datasets for arguments predicator
		# ls -d arguments_extractor/learning_process/data/shuffled_wiki_files/*.parsed | parallel --jobs 7 --u "python -m arguments_extractor -datasets -args"
		create_args_dataset(sys.argv[-1], arguments_extractor)

	elif "-parse" in sys.argv:
		# export LC_CTYPE=en_US.UTF-8
		# export LC_ALL=en_US.UTF-8
		# ls -d arguments_extractor/learning_process/data/shuffled_wiki_files/*.txt | parallel --jobs 13 --u "python -m arguments_extractor -datasets -parse"
		sentences_path = sys.argv[-1]
		create_parsed_dataset(sentences_path)

	# Use this to merge all the extracted arguments
	# cat shuffled_wiki_files/*_args.csv > shuffled_wiki_sentences_args.csv

if "-train" in sys.argv:
	arguments_predictor = ArgumentsPredictor()
	nom_lexicon = Lexicon(config.LEXICON_FILE_NAME)
	arguments_predictor.train(config.ARG_DATASET_PATH, nom_lexicon)