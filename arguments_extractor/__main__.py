import sys

from arguments_extractor.lisp_to_json.lisp_to_json import lisp_to_json
from arguments_extractor.model_based.create_datasets import create_datasets
from arguments_extractor.arguments_extractor import ArgumentsExtractor
from arguments_extractor.test import test_rule_based
from arguments_extractor.utils import separate_line_print, timeit
from arguments_extractor import config

# Generation of lexicons and datasets can be forced
if "-f" in sys.argv:
	config.LOAD_DATASET = False
	config.LOAD_LEXICON = False
	config.REWRITE_TEST = True

if "-debug" in sys.argv:
	config.DEBUG = True

if "-lispToJson" in sys.argv:
	lisp_to_json(config.LEXICON_FILE_NAME)

if "-extract" in sys.argv:
	sentence = sys.argv[-1]
	test_extractor = ArgumentsExtractor(config.LEXICON_FILE_NAME)
	test_extractor.rule_based_extraction = timeit(test_extractor.rule_based_extraction)
	extractions_per_verb, extractions_per_nom = test_extractor.rule_based_extraction(sentence)

	print("--------------------------------")
	print("VERB:")
	separate_line_print(extractions_per_verb)

	print("--------------------------------")
	print("NOM:")
	separate_line_print(extractions_per_nom)

if "-test" in sys.argv:
	test_rule_based()

if "-learning" in sys.argv:
	# Creating\Loading the datasets
	create_datasets()