import os
from os.path import join

NOMLEX_NAME = "NOMLEX-2001"
NOMLEX_PLUS_NAME = "NOMLEX-plus.1.0"

# The lexicons will be in the next paths
ABSOLUTE_PATH = os.path.dirname(__file__) # The absolute path of the root module
LISP_DIR = join(ABSOLUTE_PATH, "lexicons/lisp_lexicons/")
JSON_DIR = join(ABSOLUTE_PATH, "lexicons/json_lexicons/")
PKL_DIR = join(ABSOLUTE_PATH, "lexicons/pkl_lexicons/")

# The test files will be in the next paths
TEST_DIR = join(ABSOLUTE_PATH, "test")
TEST_SENTENCES_PATH = join(TEST_DIR, "test_sentences.txt")
TEST_VERB_EXTRACTIONS = join(TEST_DIR, "test_sentences.verb")
TEST_NOM_EXTRACTIONS = join(TEST_DIR, "test_sentences.nom")

# The Wikipedia data (parsed) files will be in the next paths
DATA_DIR = join(ABSOLUTE_PATH, "model_based/data/too_clean_wiki")
WIKI_SENTENCES_PATH = join(DATA_DIR, "shuf_wiki.txt")
EXAMPLE_SENTENCES_PATH = join(DATA_DIR, "example.txt") # sentences that are used as examples in the demo

# The dataset files will be in the next paths
DATASETS_PATH = join(ABSOLUTE_PATH, "model_based/datasets")
ARG_DATASET_PATH = join(DATASETS_PATH, "args_dataset.csv")
NOUN_DATASET_PATH = join(DATASETS_PATH, "nouns_dataset.csv")

# More configuration constants
LEXICON_FILE_NAME = NOMLEX_PLUS_NAME + ".txt"
LOAD_LEXICON = True
LOAD_DATASET = True
REWRITE_TEST = False
IGNORE_PROCESSED_DATASET = True
DEBUG = False