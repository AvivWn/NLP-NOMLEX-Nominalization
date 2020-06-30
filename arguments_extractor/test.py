import pickle
from os.path import isfile

from arguments_extractor.arguments_extractor import ArgumentsExtractor
from arguments_extractor import config

def compare_extractions(extractions, loaded_extractions, sentence):
	difference_predicates = []

	for predicate in extractions:
		if predicate not in loaded_extractions.keys() or loaded_extractions[predicate] != extractions[predicate]:
			difference_predicates.append(predicate)

	for predicate in loaded_extractions:
		if predicate not in extractions.keys() or loaded_extractions[predicate] != extractions[predicate]:
			difference_predicates.append(predicate)

	for predicate in difference_predicates:
		print(f"The extractions are different for \"{predicate}\" in the sentence\"{sentence}\"")
		print(f"OLD:{loaded_extractions[predicate]}")
		print(f"NEW:{extractions[predicate]}")

	return difference_predicates == []

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

def test_rule_based():
	"""
	Tests the rule-based extractor on a specific test set
	:return: None
	"""

	with open(config.TEST_SENTENCES_PATH, "r") as test_file:
		test_sentences = test_file.readlines()

	loaded_verb_extractions = load_extractions(config.TEST_VERB_EXTRACTIONS)
	loaded_nom_extractions = load_extractions(config.TEST_NOM_EXTRACTIONS)

	test_extractor = ArgumentsExtractor(config.LEXICON_FILE_NAME)
	found_differences = False

	for line in test_sentences:
		line = line.replace("\n", "").replace("\r", "")

		if line == "" or line.startswith("#"):
			continue

		sentence = line
		verb_extractions, nom_extractions = test_extractor.rule_based_extraction(sentence)
		verb_extractions = test_extractor.extractions_as_str(verb_extractions)
		nom_extractions = test_extractor.extractions_as_str(nom_extractions)

		if config.REWRITE_TEST:
			loaded_verb_extractions[sentence] = verb_extractions
			loaded_nom_extractions[sentence] = nom_extractions

		found_verb_differences = compare_extractions(verb_extractions, loaded_verb_extractions[sentence], sentence)
		found_nom_differences = compare_extractions(nom_extractions, loaded_nom_extractions[sentence], sentence)

		found_differences = found_verb_differences or found_nom_differences or found_differences

	if config.REWRITE_TEST:
		save_extractions(config.TEST_VERB_EXTRACTIONS, loaded_verb_extractions)
		save_extractions(config.TEST_NOM_EXTRACTIONS, loaded_nom_extractions)

	if config.REWRITE_TEST:
		print("Test Override")
	elif found_differences:
		print("Test Succesful")
	else:
		print("Test Failed")