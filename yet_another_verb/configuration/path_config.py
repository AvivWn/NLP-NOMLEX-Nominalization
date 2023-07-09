from os.path import dirname

import yet_another_verb
from yet_another_verb import constants
from yet_another_verb import nomlex

PACKAGE_PATH = dirname(yet_another_verb.__file__)
PROJECT_PATH = dirname(PACKAGE_PATH)
OUTER_PROJECT_PATH = dirname(PROJECT_PATH)

CONSTANTS_PATH = dirname(constants.__file__)
NOMLEX_PATH = dirname(nomlex.__file__)
