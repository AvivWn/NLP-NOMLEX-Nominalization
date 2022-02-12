from os.path import join, dirname

from yet_another_verb import nomlex
from yet_another_verb.nomlex.nomlex_version import NomlexVersion


class ExtractorsConfig:
	def __init__(
			self,
			extractor="nomlex",
			version=NomlexVersion.V2,
			use_nomlex_cache=True
	):
		self.EXTRACTOR = extractor

		# nomlex related configurations
		self.USE_NOMLEX_CACHE = use_nomlex_cache
		self.NOMLEX_VERSION = version
		self.LEXICON_DIR = f"{dirname(nomlex.__file__)}/lexicons"
		self.LISP_LEXICON_DIR = join(self.LEXICON_DIR, "lisp")
		self.JSON_LEXICON_DIR = join(self.LEXICON_DIR, "json")
		self.PKL_LEXICON_DIR = join(self.LEXICON_DIR, "pkl")


EXTRACTORS_CONFIG = ExtractorsConfig()
