from typing import List
from copy import deepcopy
from more_itertools import unique_everseen

from noun_as_verb.nomlex_representation.lexicon_constants import LexiconTag, NomTypeProperty, EntryProperty
from noun_as_verb.nomlex_representation.lexicon_constants import SubcatProperty


def _filter_subcats_by_property(entry: dict, filter_property: str) -> List[str]:
	filtered_subcats = []

	for subcat_type, subcat in entry[EntryProperty.SUBCATS].items():
		if filter_property in subcat.keys():
			filtered_subcats.append(subcat_type)

	return filtered_subcats


def devide_by_nom_types(entry: dict) -> List[dict]:
	"""
	Devides the given entry to multiple entries for each nom type
	:param entry: a nomlex entry as a json format
	:return: the devided lexicon entries
	"""

	if len(entry[EntryProperty.NOM_TYPE]) <= 1:
		return [entry]

	new_entries = []
	for nom_type_info in entry[EntryProperty.NOM_TYPE]:
		new_entry = deepcopy(entry)
		new_entry[EntryProperty.NOM_TYPE] = [nom_type_info]
		new_entries.append(new_entry)

	return new_entries


def devide_by_alternates_opt(entry: dict) -> List[dict]:
	"""
	Devides the given entry based on the tag ALTERNATES-OPT (optional alternation)
	Two entries are generated if the tag is specified (with and without alternations)
	:param entry: a nomlex entry as a json format
	:return: the devided lexicon entries
	"""

	# Check whether the feature ALTERNATES-OPT appears in any of the nom subcats
	subcats_with_alternates_opt = _filter_subcats_by_property(entry, LexiconTag.ALTERNATES_OPT)
	if len(subcats_with_alternates_opt) == 0:
		return [entry]

	alternates_entry = deepcopy(entry)
	alternates_entry[EntryProperty.SUBCATS] = {}

	no_alternates_entry = deepcopy(entry)
	no_alternates_entry[EntryProperty.FEATURES].pop(LexiconTag.SUBJ_IND_OBJ_ALT, None).pop(LexiconTag.SUBJ_OBJ_ALT, None)

	for subcat_type in subcats_with_alternates_opt:
		# ALTERNATES-OPT -> ALTERNATES
		alternates_subcat = alternates_entry[EntryProperty.SUBCATS][subcat_type] = deepcopy(
			entry[EntryProperty.SUBCATS][subcat_type])
		alternates_subcat[SubcatProperty.ALTERNATES] = alternates_subcat.pop(LexiconTag.ALTERNATES_OPT)

		# without ALTERNATES and ALTERNATES-OPT
		no_alternates_entry[EntryProperty.SUBCATS][subcat_type].pop(LexiconTag.ALTERNATES_OPT)

	return [alternates_entry, no_alternates_entry]


def devide_by_adval_nom(entry: dict) -> List[dict]:
	"""
	Devides the given entry based on the tag ADVAL-NOM
	Each ADVAL-NOM value for each subcat generates another entry
	:param entry: a nomlex entry as a json format
	:return: the devided lexicon entries
	"""

	# Check whether the feature ADVAL-NOM appears in any of the nom subcats
	subcats_with_adval_nom = _filter_subcats_by_property(entry, LexiconTag.ADVAL_NOM)
	if len(subcats_with_adval_nom) == 0:
		return [entry]

	no_adval_nom_entry = deepcopy(entry)
	new_entries = [no_adval_nom_entry]

	for subcat_type in subcats_with_adval_nom:
		nom_subcat = entry[EntryProperty.SUBCATS][subcat_type]

		# Create new nominalization for any particle that can appear with the nom
		for particle in nom_subcat[LexiconTag.ADVAL_NOM]:
			new_entry = deepcopy(entry)
			new_entry[EntryProperty.ORTH] = entry[EntryProperty.ORTH] + "-" + particle
			new_entry[EntryProperty.SUBCATS] = {}
			new_entry[EntryProperty.NOM_TYPE] = []

			# Updating the type of the new nom according to the particle
			for nom_type_info in deepcopy(entry[EntryProperty.NOM_TYPE]):
				nom_type = nom_type_info[NomTypeProperty.TYPE]
				assert "-PART" not in nom_type
				nom_type_info[NomTypeProperty.TYPE] = nom_type.replace("-NOM", "") + "-PART"
				nom_type_info[NomTypeProperty.PART] = particle
				new_entry[EntryProperty.NOM_TYPE].append(nom_type_info)

			new_subcat = new_entry[EntryProperty.SUBCATS][subcat_type] = deepcopy(nom_subcat)
			new_subcat.pop(LexiconTag.ADVAL_NOM)
			new_subcat[LexiconTag.ADVAL] = [particle]
			new_entries.append(new_entry)

		no_adval_nom_entry[EntryProperty.SUBCATS].pop(subcat_type)

	return new_entries


def duplicate_by_hyphen(entry: dict) -> List[dict]:
	"""
	Duplicates the given entry when hyphen is optional
	Add the hyphen when missing, and remove it when appearing
	:param entry: a nomlex entry as a json format
	:return: the duplicated lexicon entries
	"""

	orth = entry[EntryProperty.ORTH]

	if "-" in orth:
		# Generate entry without hyphen (come-back -> comeback)
		no_hyphen_entry = deepcopy(entry)
		no_hyphen_entry[EntryProperty.ORTH] = orth.replace("-", "")
		return [entry, no_hyphen_entry]

	particle = entry[EntryProperty.NOM_TYPE][0].get(NomTypeProperty.PART, None)
	if particle is None:
		return [entry]

	# Generate entry with hyphen (comeback -> come-back, output -> out-put)
	hyphen_entry = deepcopy(entry)
	hyphen_entry[EntryProperty.ORTH] = orth.replace(particle, "-" + particle).replace(particle, particle + "-").strip("-")

	if hyphen_entry[EntryProperty.ORTH] != orth:
		return [entry, hyphen_entry]

	return [entry]


def devide_to_entries(entry: dict) -> List[dict]:
	"""
	Returns the devision of the given entry into multiple entries
	:param entry: a nomlex entry as a json format
	:return: the devided lexicon entries
	"""

	last_related_entries = []
	new_related_entries = devide_by_nom_types(entry)

	# Stop when there aren't any new entries
	while last_related_entries != new_related_entries:
		last_related_entries = new_related_entries
		new_related_entries = []

		for entry in last_related_entries:
			new_related_entries += \
				devide_by_alternates_opt(entry) + \
				devide_by_adval_nom(entry) + \
				duplicate_by_hyphen(entry)

		new_related_entries = list(unique_everseen(new_related_entries))

	return last_related_entries
