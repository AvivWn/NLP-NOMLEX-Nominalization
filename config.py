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

LISP_DIR = "LispLexicons/"
JSON_DIR = "JsonLexicons/"
PKL_DIR = "PklLexicons/"
LOAD_LEXICON = True
DEBUG = False