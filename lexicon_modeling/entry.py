from .subcat import *

class Entry:
	orth: str
	features: dict
	subcats = defaultdict(Subcat)
	next = None
	plural = None
	nom = None
	singular = None
	plural_freq = None
	is_verb: bool



	def __init__(self, entry: dict, is_verb):
		self.orth = entry[ENT_ORTH]
		self.features = entry.get(ENT_FEATURES, {})
		self.nom_type = entry.get(ENT_NOM_TYPE, {})

		self.subcats = defaultdict(Subcat)
		for subcat_type in entry.get(ENT_VERB_SUBC, {}).keys():
			self.subcats[subcat_type] = Subcat(entry[ENT_VERB_SUBC][subcat_type], is_verb)

		self.next = entry.get(ENT_NEXT, None)
		self.plural = entry.get(ENT_PLURAL, None)
		self.nom = entry.get(ENT_NOM, None)
		self.singular = entry.get(ENT_SINGULAR, None)
		self.plural_freq = entry.get(ENT_PLURAL_FREQ, None)
		self.is_verb = is_verb

	def set_next(self, lexicon):
		self.next = lexicon.get_entry(self.next)

	def extract_arguments(self, dependency_tree: list, argument_candidates: list):
		extractions = []

		# Extract the arguments based on each subcat for this word entry
		for subcat_type in self.subcats.keys():
			print(subcat_type)
			extractions += self.subcats[subcat_type].extract_arguments(dependency_tree, argument_candidates)

		# Extract arguments also based on the "next" entry in the lexicon
		# Meaning by the same word with another sense
		if self.next is not None:
			extractions += self.next.extract_arguments(dependency_tree, argument_candidates)

		extractions = sorted(extractions, key=lambda k: len(k.keys()), reverse=True)

		final_extractions = []
		for extraction in deepcopy(extractions):
			is_sub_extraction = any([extraction.items() <= other_extraction.items() for other_extraction in final_extractions])

			if not is_sub_extraction:
				final_extractions.append(extraction)

		return final_extractions