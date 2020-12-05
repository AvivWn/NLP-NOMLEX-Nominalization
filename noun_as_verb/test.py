import pickle
from os.path import isfile

from tqdm import tqdm

from noun_as_verb.arguments_extractor import ArgumentsExtractor
from noun_as_verb import config

def compare_extractions(extractions, loaded_extractions, sentence):
	difference_predicates = []

	dict_key_func = lambda d: sorted((k, v if v is not None else '') for k, v in d.items())

	for predicate in extractions:
		if predicate not in loaded_extractions.keys() or sorted(loaded_extractions[predicate], key=dict_key_func) != sorted(extractions[predicate], key=dict_key_func):
			difference_predicates.append(predicate)

	for predicate in loaded_extractions:
		if predicate not in extractions.keys():
			difference_predicates.append(predicate)

	for predicate in difference_predicates:
		print(f"The extractions are different for \"{predicate}\" in the sentence: \"{sentence}\"")
		print(f"OLD: {loaded_extractions.get(predicate, 'NONE')}")
		print(f"NEW: {extractions.get(predicate, 'NONE')}")

	return difference_predicates != []

def load_extractions(extractions_file_path):
	loaded_extractions = {}
	if isfile(extractions_file_path) and not config.REWRITE_TEST:
		with open(extractions_file_path, "rb") as loaded_extractions:
			loaded_extractions = pickle.load(loaded_extractions)
	else:
		config.REWRITE_TEST = True

	return loaded_extractions

def save_extractions(extractions_file_path, extractions):
	with open(extractions_file_path, "wb") as extractions_file:
		pickle.dump(extractions, extractions_file)

def test(arguments_extractor: ArgumentsExtractor, extraction_func):
	"""
	Tests the given extraction function vs saved rule-based extraction of a specific test set
	:param arguments_extractor: The extractor object, which receives a sentence and returns all its extractions
	:param extraction_func: The extraction function that we should use on the test examples
	:return: None
	"""

	with open(config.TEST_SENTENCES_PATH, "r") as test_file:
		test_sentences = test_file.readlines()

	loaded_verb_extractions = load_extractions(config.TEST_VERB_EXTRACTIONS)
	loaded_nom_extractions = load_extractions(config.TEST_NOM_EXTRACTIONS)

	# Only rule-based extractions should be saved in a file
	if extraction_func != ArgumentsExtractor.rule_based_extraction:
		config.REWRITE_TEST = False

	found_differences = False

	for line in tqdm(test_sentences, "Testing", leave=False):
		line = line.strip(" \t\n\r")

		if line == "" or line.startswith("#"):
			continue

		sentence = line.split("#")[0].strip()
		verb_extractions, nom_extractions = extraction_func(arguments_extractor, sentence)
		verb_extractions = arguments_extractor.extractions_as_str(verb_extractions)
		nom_extractions = arguments_extractor.extractions_as_str(nom_extractions)

		if config.REWRITE_TEST:
			loaded_verb_extractions[sentence] = verb_extractions
			loaded_nom_extractions[sentence] = nom_extractions

		found_verb_differences = compare_extractions(verb_extractions, loaded_verb_extractions.get(sentence, {}), sentence)
		found_nom_differences = compare_extractions(nom_extractions, loaded_nom_extractions.get(sentence, {}), sentence)

		found_differences = found_verb_differences or found_nom_differences or found_differences

	if config.REWRITE_TEST:
		save_extractions(config.TEST_VERB_EXTRACTIONS, loaded_verb_extractions)
		save_extractions(config.TEST_NOM_EXTRACTIONS, loaded_nom_extractions)

	if config.REWRITE_TEST:
		print("Test Overrided!")
	elif not found_differences:
		print("Test Succeeded!")
	else:
		print("Test Failed!")