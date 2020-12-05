from copy import deepcopy
from tqdm import tqdm

from .utils import get_current_specs, curr_specs
from noun_as_verb.constants.lexicon_constants import *
from noun_as_verb.utils import engine

def duplicate_entry(lexicon, entry, new_orth=None):
	"""
	Duplicates the given entry and names it as the given new orth (if given) or as the orth of the given entry
	The function also adds the new entry to the given lexicon, without colliding two different entries
	:param lexicon: a nomlex lexicon as a json format
	:param entry: an entry in the lexicon as a dictionary of dictionaries
	:param new_orth: the wanted orth for the new duplicated entry
	:return: the string of the nom that was eventually added to the lexicon
	"""

	if new_orth is None:
		new_orth = entry[ENT_ORTH]

	# Generate any nom with the wanted orth
	new_numbered_orth = new_orth

	if new_orth in lexicon.keys():
		count = 2
		while new_orth + "#" + str(count) in lexicon.keys():
			count += 1

		new_numbered_orth += "#" + str(count)

	# The new nom entry will be the same as the given entry
	lexicon[new_numbered_orth] = deepcopy(entry)

	# The orth will be as the wanted one, and also the plural form will be influenced
	lexicon[new_numbered_orth][ENT_ORTH] = new_orth

	# Update the plural form if it existed in the given entry
	if entry.get(ENT_PLURAL, "NONE") not in NONE_VALUES:
		lexicon[new_numbered_orth][ENT_PLURAL] = engine.plural(new_orth)

	return new_numbered_orth

def split_alternates_opt(lexicon, nom):
	"""
	Splits the entry of the given nom if it includes the tag ALTERNATES-OPT
	The original nom entry will includes a standard alternation, and the new one won't include any
	:param lexicon: a nomlex lexicon as a json format
	:param nom: a specific nominalization key in the lexicon
	:return: a list of the nominalizations that were added to the lexicon, based on the one given nom and on this condition
	"""

	nom_entry = lexicon[nom]
	nom_subcats = deepcopy(lexicon[nom]).get(ENT_VERB_SUBC, {}).items()

	# Check whether the feature ALTERNATES-OPT appears in any of the nom subcats
	subcats_with_alternates_opt = [subcat_type for subcat_type, subcat in nom_subcats if OLD_SUBCAT_CONSTRAINT_ALTERNATES_OPT in subcat.keys()]
	if subcats_with_alternates_opt == []:
		return []

	new_nom = duplicate_entry(lexicon, nom_entry)
	new_nom_entry = lexicon[new_nom]
	new_nom_entry[ENT_FEATURES].pop(FEATURE_SUBJ_IND_OBJ_ALT, None).pop(FEATURE_SUBJ_OBJ_ALT, None)

	# The new nom will include only the subcats that included ALTERNATES-OPT, but now they won't include any alternation
	new_nom_entry[ENT_VERB_SUBC] = {}

	# Now any ALTERNATES-OPT is exactly ALTERNATES and in the other entry there isn't any ALTERNATES tag
	# The other subcats stay the same in both
	for subcat_type in subcats_with_alternates_opt:
		new_nom_subcat = new_nom_entry[ENT_VERB_SUBC][subcat_type] = deepcopy(nom_entry[ENT_VERB_SUBC][subcat_type])

		# Original nom: ALTERNATES-OPT -> ALTERNATES
		nom_subcat = nom_entry[ENT_VERB_SUBC][subcat_type]
		nom_subcat[SUBCAT_CONSTRAINT_ALTERNATES] = nom_subcat.pop(OLD_SUBCAT_CONSTRAINT_ALTERNATES_OPT)

		# New nom: without ALTERNATES and ALTERNATES-OPT
		new_nom_subcat.pop(OLD_SUBCAT_CONSTRAINT_ALTERNATES_OPT)

	return [new_nom]

def split_adval_nom(lexicon, nom):
	"""
	Splits the entry of the given nom if it includes any subact with the ADVAL-NOM values
	Each value of ADVAL-NOM (= each particle) on any subcat will create new nom entry
	:param lexicon: a nomlex lexicon as a json format
	:param nom: a specific nominalization key in the lexicon
	:return: a list of the nominalizations that were added to the lexicon, based on the one given nom and on this condition
	"""

	nom_entry = lexicon[nom]
	nom_subcats = lexicon[nom].get(ENT_VERB_SUBC, {}).items()

	# Check whether the feature ADVAL-NOM appears in any of the nom subcats
	subcats_with_adval_nom = [subcat_type for subcat_type, subcat in nom_subcats if OLD_COMP_ADVAL_NOM in subcat.keys()]
	if subcats_with_adval_nom == []:
		return []

	# By now the nominalizations should have only one type
	nom_type = list(nom_entry[ENT_NOM_TYPE].keys())[0]

	# Assumption- a nominalization with NOM-ADVAL cannot be a PART-typed nom (because otherwise the ADVAL-NOM values don't have any meaning)
	if "-PART" in nom_type:
		raise Exception(f"PART-typed nom cannot specify any ADVAL-NOM values ({get_current_specs()}).")

	# The type of any new nominalizaiton will include the PART word in the end
	new_nom_type = nom_type.replace("-NOM", "") + "-PART"
	new_noms = []

	# Search again for the subcats with ADVAL-NOM
	for subcat_type in subcats_with_adval_nom:
		nom_subcat = nom_entry[ENT_VERB_SUBC][subcat_type]

		# Create new nominalization for any particle that can appear with the nom
		for particle in nom_subcat[OLD_COMP_ADVAL_NOM]:
			# The new nominalization with the particle
			nom_with_part = nom_entry[ENT_ORTH] + "-" + particle

			# Don't create the new nominalization if it has already appeared in the lexicon
			if nom_with_part in lexicon.keys():
				new_nom_entry = lexicon[nom_with_part]
			else:
				new_nom = duplicate_entry(lexicon, nom_entry, new_orth=nom_with_part)
				new_nom_entry = lexicon[new_nom]
				new_nom_entry[ENT_VERB_SUBC] = {}

			# Updating the type of the new nom according to the particle
			new_nom_entry[ENT_NOM_TYPE][new_nom_type] = new_nom_entry[ENT_NOM_TYPE].pop(nom_type)
			new_nom_entry[ENT_NOM_TYPE][new_nom_type].update({OLD_COMP_ADVAL:[particle]})

			# The new nom will get only the current subcat cause it is the only one it can appear in
			new_nom_entry[ENT_VERB_SUBC].update({subcat_type: deepcopy(nom_subcat)})
			new_nom_subcat = new_nom_entry[ENT_VERB_SUBC][subcat_type]
			new_nom_subcat[OLD_COMP_ADVAL] = new_nom_subcat.pop(OLD_COMP_ADVAL_NOM)

			new_noms.append(nom_with_part)

		# The subcats with ADVAL-NOM will be removed from the original nom
		# Cause it is was transfered for the new nom
		nom_entry[ENT_VERB_SUBC].pop(subcat_type)

	return new_noms

def split_nom_types(lexicon, nom):
	"""
	Splits the entry of the given nom if it includes more than one nom type
	:param lexicon: a nomlex lexicon as a json format
	:param nom: a specific nominalization key in the lexicon
	:return: a list of the nominalizations that were added to the lexicon, based on the one given nom and on this condition
	"""

	nom_entry = lexicon[nom]

	# Split nominalization types when there is more than one nom type
	if len(nom_entry[ENT_NOM_TYPE].keys()) <= 1:
		return []

	new_noms = []
	nom_types = list(deepcopy(nom_entry[ENT_NOM_TYPE]).items())
	first_nom_type = nom_types[0][0]
	nom_entry[ENT_NOM_TYPE] = {first_nom_type: nom_entry[ENT_NOM_TYPE][first_nom_type]}

	# Creating new nominalization for any possible nom type
	for nom_type, nom_type_value in nom_types:
		if nom_type == first_nom_type:
			continue

		new_nom = duplicate_entry(lexicon, nom_entry)
		new_entry = lexicon[new_nom]
		new_entry[ENT_NOM_TYPE] = {nom_type: nom_type_value}
		new_noms.append(new_nom)

	return new_noms

def update_particles(entry):
	"""
	Extract the specified possible particles for the given entry
	If the entry is suitable for a PART-typed nom and the particle isn't given, then it is extracted from the nom itself
	In both cases, the particles list is updated according the founded possible particles
	:param entry: an entry in the lexicon as a dictionary of dictionaries
	:return: a list of the founded possible particles for the given entry
	"""

	# By now the entry should includes only one nom type
	nom_type = list(entry[ENT_NOM_TYPE].keys())[0]
	nom_value = entry[ENT_NOM_TYPE][nom_type]

	particles = []

	# Only noms with PART in their type can get particles
	if "PART" in nom_type:
		# Extracting the particle from the nom itseft by comapring to the verb
		if OLD_COMP_ADVAL not in entry[ENT_NOM_TYPE][nom_type]:
			particles = [entry[ENT_ORTH].replace(entry[ENT_VERB], "").replace("-", "")]

		# Extracting the particle from the nom-type
		else:
			particles = nom_value.get(OLD_COMP_ADVAL, [])
			if type(particles) == str: particles = [particles]

		entry[ENT_NOM_TYPE][nom_type][OLD_COMP_ADVAL] = particles

		# There should be only one particle for each PART-typed nom
		if len(particles) != 1:
			raise Exception(f"The PART-typed nominalizations should have exactly one appropriate particle ({get_current_specs()}).")

	return particles

def duplicate_nom_hyphen(lexicon, nom):
	"""
	Duplicates the entry of the given nom if it includes hyphen or should include one
	:param lexicon: a nomlex lexicon as a json format
	:param nom: a specific nominalization key in the lexicon
	:return: a list of the nominalizations that were added to the lexicon, based on the one given nom and on this condition
	"""

	nom_entry = lexicon[nom]
	nom_orth = nom_entry[ENT_ORTH]

	new_noms = []

	# Update the particle in the nom-type of the nominalization
	# Only for nominalization that take the particle of the verb
	particles = update_particles(nom_entry)

	# A hyphen may appear in the nom (like come-back or co-operation)
	if "-" in nom:
		# Multiply the nom entry only if it hasn't appeared in the lexicon already (with the same subcats)
		new_orth = nom_orth.replace("-", "")
		tmp_nom_entry = deepcopy(nom_entry)
		tmp_nom_entry[ENT_ORTH] = new_orth

		for curr_nom_entry in lexicon.values():
			if curr_nom_entry == tmp_nom_entry:
				return []

		# Add to the lexicon, the suitable nom without the hyphen
		return [duplicate_entry(lexicon, nom_entry, new_orth=new_orth)]

	# Otherwise, if the hyphen doesn't appear in the nom, then the contiguous version may appear in the lexicon
	# We cares only for a particular case of optional hyphen- between the nom verb-particle (comeback)

	if particles == []:
		return []
	particle = particles[0]

	# Create the new nom with the particle and a hypen. There can be several positions for the particle

	# The particle appear after the nom (come-back)
	if nom_orth.endswith(particle):
		new_orth = nom_orth.replace("-", "-" + particle)

	# The particle appear before the nom (out-put)
	elif nom_orth.startswith(particle):
		new_orth = nom_orth.replace(particle, particle + "-")

	# The particle doesn't appear with its particle (probably by mistake)
	else:
		new_nom = duplicate_entry(lexicon, nom_entry, new_orth=nom_orth + "-" + particle)
		lexicon.pop(nom)
		return duplicate_nom_hyphen(lexicon, new_nom)

	tmp_nom_entry = deepcopy(nom_entry)
	tmp_nom_entry[ENT_ORTH] = new_orth

	for curr_nom_entry in lexicon.values():
		if curr_nom_entry == tmp_nom_entry:
			return []

	new_nom = duplicate_entry(lexicon, nom_entry, new_orth=new_orth)
	new_noms.append(new_nom)

	return new_noms

def split_entry(lexicon, nom):
	"""
	Splits the entry of the given nom on specific conditions
	:param lexicon: a nomlex lexicon as a json format
	:param nom: a specific nominalization key in the lexicon
	:return: a list of the nominalizations that were added to the lexicon, based on the one given nom
	"""

	new_noms = []

	# Noms that includes the tag ALTERNATES-OPT should be splitted into two different entries
	# One entry with ALTERNATES and the other one without ALTERNATES
	new_noms += split_alternates_opt(lexicon, nom)

	# Noms with multiple nom types should be splitted such that any entry include only one type
	new_noms += split_nom_types(lexicon, nom)

	# Noms that includes the tag ADVAL-NOM should be splitted into two different entries
	# One entry for the nom with the particle and the other one for just the nom
	new_noms += split_adval_nom(lexicon, nom)

	# Noms with particles (like backup) should appear in the lexicon with hyphen ('-') and without it
	# Other noms with hyphen (like co-operation) should appear in the lexicon also without it
	new_noms += duplicate_nom_hyphen(lexicon, nom)

	# Split again the new added entries
	for new_nom in new_noms:
		new_noms += split_entry(lexicon, new_nom)

	return new_noms

def split_entries(lexicon):
	"""
	Splits entries in the lexicon that acts like too different entries
	:param lexicon: a nomlex lexicon as a json format
	:return: a list of the nominalizations that were added to the lexicon
	"""

	new_noms = []

	for nom in tqdm(deepcopy(lexicon).keys(), "Splitting some entries", leave=False):
		curr_specs["word"] = lexicon[nom][ENT_ORTH]
		new_noms += split_entry(lexicon, nom)

	return new_noms



def remove_entries(lexicon):
	"""
	Removes entries in the lexicon that are considered to have mistakes/errors
	:param lexicon: a nomlex lexicon as a json format
	:return: a list of the nominalization that were removed from the lexicon
	"""

	removed_noms = []
	tmp_lexicon = deepcopy(lexicon)

	for nom in tqdm(tmp_lexicon.keys(), "Removing mistaken entries", leave=False):
		if nom not in lexicon:
			continue

		nom_entry = lexicon[nom]

		# Remove all the entries of a nom that have multiple appropriated verbs
		found_another_verb = False
		for other_nom in tmp_lexicon.keys():
			other_verb = tmp_lexicon[other_nom][ENT_VERB]
			if other_nom.split("#")[0] == nom.split("#")[0] and other_verb != nom_entry[ENT_VERB]:
				found_another_verb = True
				removed_noms.append(other_nom)
				lexicon.pop(other_nom)

		if found_another_verb:
			removed_noms.append(nom)
			lexicon.pop(nom)
			continue

		# The entry must include the ORTH tag
		if ENT_ORTH not in nom_entry.keys():
			raise Exception(f"Any nom entry should specify the orth under ORTH (nom={nom})")

		curr_specs["word"] = nom_entry[ENT_ORTH]
		nom_features = nom_entry.get(ENT_FEATURES, {}).keys()
		nom_subcats = nom_entry.get(ENT_VERB_SUBC, {}).values()

		# Removing nominalizations without nom type property
		if ENT_NOM_TYPE not in nom_entry.keys():
			removed_noms.append(nom)
			lexicon.pop(nom)
			continue

		# ALTERNATES(-OPT) tags goes together with one of the features SUBJ-OBJ-ALT or SUBJ-IND-OBJ-ALT
		# Check whether or not these features and the ALTERNATES(-OPT) tag appears for this nominalization
		alt_feature_appear = FEATURE_SUBJ_OBJ_ALT in nom_features or FEATURE_SUBJ_IND_OBJ_ALT in nom_features
		any_alternates_appear = any([SUBCAT_CONSTRAINT_ALTERNATES in subcat.keys() or OLD_SUBCAT_CONSTRAINT_ALTERNATES_OPT in subcat.keys() for subcat in nom_subcats])

		# Assuming that nominalizations with ALTERNATES tags and without any of those features aren't correct
		if not alt_feature_appear and any_alternates_appear:
			removed_noms.append(nom)
			lexicon.pop(nom)

	return removed_noms