from copy import deepcopy
from collections import defaultdict
from itertools import product
from tqdm import tqdm
import numpy as np
import pickle
import json
import time
import os
import re
import inflect
engine = inflect.engine()

# The lexicons are assumed to be in the next relative paths
LISP_DIR = "lexicons/lisp_lexicons/"
JSON_DIR = "lexicons/json_lexicons/"
PKL_DIR = "lexicons/pkl_lexicons/"

# More configuration constants
LOAD_LEXICON = True
DEBUG = False