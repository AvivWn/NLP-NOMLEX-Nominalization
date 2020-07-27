import os

# The lexicons will be in the next paths
ABSOLUTE_PATH = os.path.dirname(__file__) # The absolute path of the root module
LISP_DIR = ABSOLUTE_PATH + "/lexicons/lisp_lexicons/"
JSON_DIR = ABSOLUTE_PATH + "/lexicons/json_lexicons/"
PKL_DIR = ABSOLUTE_PATH + "/lexicons/pkl_lexicons/"

# The test files will be in the next paths
TEST_DIR = ABSOLUTE_PATH + "/test"
TEST_SENTENCES_PATH = TEST_DIR + "/test_sentences.txt"
TEST_VERB_EXTRACTIONS = TEST_DIR + "/test_sentences.verb"
TEST_NOM_EXTRACTIONS = TEST_DIR + "/test_sentences.nom"

# The Wikipedia data (parsed) files will be in the next paths
DATA_DIR = ABSOLUTE_PATH + "/learning_process/data"
WIKI_SENTENCES_PATH = DATA_DIR + "/shuffled_wiki_sentences.txt"
EXAMPLE_SENTENCES_PATH = DATA_DIR + "/example_sentences.txt"	# sentences that are used as examples in the demo
ARG_SENTENCES_PATH = DATA_DIR + "/arguments_sentences.txt"		# sentences that were used to create arguments dataset

# The dataset files will be in the next paths
ARG_DATASET_DIR = ABSOLUTE_PATH + "/model_based/datasets"

# More configuration constants
LEXICON_FILE_NAME = "NOMLEX-plus.1.0.txt"
LOAD_LEXICON = True
LOAD_DATASET = True
REWRITE_TEST = False
DEBUG = False
CLEAN_NP = True