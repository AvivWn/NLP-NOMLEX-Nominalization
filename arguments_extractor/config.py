import os

# The lexicons are assumed to be in the next relative paths
ABSOLUTE_PATH = os.path.dirname(__file__) # The absolute path of the root module
LISP_DIR = ABSOLUTE_PATH + "/lexicons/lisp_lexicons/"
JSON_DIR = ABSOLUTE_PATH + "/lexicons/json_lexicons/"
PKL_DIR = ABSOLUTE_PATH + "/lexicons/pkl_lexicons/"

TEST_DIR = ABSOLUTE_PATH + "/test"
TEST_SENTENCES_PATH = TEST_DIR + "/test_sentences"
TEST_VERB_EXTRACTIONS = TEST_DIR + "/test_sentences.verb"
TEST_NOM_EXTRACTIONS = TEST_DIR + "/test_sentences.nom"


DATA_DIR = ABSOLUTE_PATH + "/model_based/data"
WIKI_SENTENCES_PATH = DATA_DIR + "/shuffled_wiki_sentences"
SENTENCES_PATH = DATA_DIR + "/sentences"


# WIKIPEDIA_DIR = DATA_DIR + "/wikipedia_files"
# PARSED_SENTENCES_PATH = DATA_DIR + "/sentences.parsed"

DATASET_DIR = ABSOLUTE_PATH + "/model_based/datasets"
TRAIN_SET_PATH = DATASET_DIR + "/train"
TEST_SET_PATH = DATASET_DIR + "/test"

# More configuration constants
LEXICON_FILE_NAME = "NOMLEX-plus.1.0.txt"
LOAD_LEXICON = True
LOAD_DATASET = True
REWRITE_TEST = False
DEBUG = False