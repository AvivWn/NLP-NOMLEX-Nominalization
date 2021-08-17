from os.path import join

from yet_another_verb.nomlex.nomlex_version import NomlexVersion


class NomlexConfig:
	def __init__(
			self,
			version=NomlexVersion.V2,
			use_cache=True,
	):
		self.USE_CACHE = use_cache

		self.NOMLEX_VERSION = version
		self.LEXICON_DIR = "yet_another_verb/nomlex/lexicons"
		self.LISP_LEXICON_DIR = join(self.LEXICON_DIR, "lisp")
		self.JSON_LEXICON_DIR = join(self.LEXICON_DIR, "json")
		self.PKL_LEXICON_DIR = join(self.LEXICON_DIR, "pkl")


NOMLEX_CONFIG = NomlexConfig()
