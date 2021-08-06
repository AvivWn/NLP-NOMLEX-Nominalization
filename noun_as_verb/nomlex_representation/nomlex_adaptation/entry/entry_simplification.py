from noun_as_verb.nomlex_representation.lexicon_constants import NomTypeProperty, EntryProperty


def flatten_entry(entry: dict):
	assert len(entry.keys()) == 1
	entry_type, entry_info = list(entry.items())[0]
	entry.pop(entry_type)
	entry.update(entry_info)
	entry[EntryProperty.TYPE] = entry_type


def _get_poped_keys_at(dictionary: dict, position: str):
	return list(dictionary.pop(position, {}).keys())


def simplify_properties(entry: dict):
	entry[EntryProperty.OBJ_ATTRIBUTE] = _get_poped_keys_at(entry, EntryProperty.OBJ_ATTRIBUTE)
	entry[EntryProperty.SUBJ_ATTRIBUTE] = _get_poped_keys_at(entry, EntryProperty.SUBJ_ATTRIBUTE)
	entry[EntryProperty.IND_OBJ_ATTRIBUTE] = _get_poped_keys_at(entry, EntryProperty.IND_OBJ_ATTRIBUTE)
	entry[EntryProperty.NOUN] = _get_poped_keys_at(entry, EntryProperty.NOUN)
	entry[EntryProperty.SEMI_AUTOMATIC] = entry.pop(EntryProperty.SEMI_AUTOMATIC, False)
	entry[EntryProperty.SUBCATS] = entry.pop(EntryProperty.VERB_SUBC, {})
	entry[EntryProperty.NOUN_SUBC] = entry.pop(EntryProperty.NOUN_SUBC, {})
	entry[EntryProperty.FEATURES] = entry.pop(EntryProperty.FEATURES, {})
	entry[EntryProperty.DET_POSS_NO_OTHER_OBJ] = _get_poped_keys_at(entry, EntryProperty.DET_POSS_NO_OTHER_OBJ)
	entry[EntryProperty.N_N_MOD_NO_OTHER_OBJ] = _get_poped_keys_at(entry, EntryProperty.N_N_MOD_NO_OTHER_OBJ)

	if EntryProperty.ORTH in entry:
		assert isinstance(entry[EntryProperty.ORTH], str)
		entry[EntryProperty.ORTH] = ''.join([i for i in entry.pop(EntryProperty.ORTH) if not i.isdigit()])


def rearrange_nom_type(entry: dict):
	original_nom_types = entry.pop(EntryProperty.NOM_TYPE, {})
	new_nom_types = []

	for nom_type, nom_type_value in original_nom_types.items():
		particles = []

		if "PART" in nom_type:
			if NomTypeProperty.ADVAL in nom_type_value:
				particles = nom_type_value.get(NomTypeProperty.ADVAL, [])
				if type(particles) == str:
					particles = [particles]
			else:
				# Extracting the particle from the nom itseft by comapring to the verb
				particle = entry[EntryProperty.ORTH].replace(entry[EntryProperty.VERB], "")
				particles = [particle.replace("-", "")]

		assert len(particles) <= 1, "There should at-most only one possible particle per nominalization"
		particle = particles[0] if len(particles) == 1 else None

		new_nom_types.append({
			NomTypeProperty.TYPE: nom_type,
			NomTypeProperty.PART: particle,
			NomTypeProperty.PVAL: nom_type_value.get(NomTypeProperty.PVAL, [])
		})

	entry[EntryProperty.NOM_TYPE] = new_nom_types


def standarize_str_properties(entry: dict):
	# Avoiding NONE and T values as strings
	for entry_property in entry.keys():
		if not isinstance(entry[entry_property], str):
			continue

		if entry[entry_property] == "T":
			entry[entry_property] = True
		elif entry[entry_property].lower() in ["none", "*none*", "none"]:
			entry[entry_property] = None


def simplify_entry(entry: dict):
	"""
	Simplify the value of every property except to the subcategorization
	:param entry: a nomlex entry as a json format
	"""

	flatten_entry(entry)
	simplify_properties(entry)
	rearrange_nom_type(entry)
	standarize_str_properties(entry)
