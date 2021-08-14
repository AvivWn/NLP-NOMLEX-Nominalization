from os.path import join, dirname

from yet_another_verb.nomlex import lexicons
from yet_another_verb.nomlex.nomlex_version import NomlexVersion

LEXICON_DIR = dirname(lexicons.__file__)


class NomlexConfig:
	def __init__(
			self,
			version=NomlexVersion.V2,
			use_cache=True,
	):
		self.USE_CACHE = use_cache

		self.NOMLEX_VERSION = version
		self.LISP_LEXICON_DIR = join(LEXICON_DIR, "lisp")
		self.JSON_LEXICON_DIR = join(LEXICON_DIR, "json")


NOMLEX_CONFIG = NomlexConfig()
