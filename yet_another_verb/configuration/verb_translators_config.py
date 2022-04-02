from os.path import join, dirname

from yet_another_verb import word_to_verb
from yet_another_verb.nomlex.nomlex_version import NomlexVersion


class VerbTranslatorsConfig:
	def __init__(
			self,
			translator="nomlex",
			version=NomlexVersion.V2,
			use_cache=True
	):
		self.TRANSLATOR = translator
		self.TRANSLATORS_CACHE_DIR = join(dirname(word_to_verb.__file__), "cache")
		self.USE_CACHE = use_cache

		# nomlex related configurations
		self.NOMLEX_VERSION = version


VERB_TRANSLATORS_CONFIG = VerbTranslatorsConfig()
