from yet_another_verb.nomlex.constants import LexiconType, NomTypeProperty, \
	EntryProperty, SubcatType, SubcatProperty, ArgumentType, ArgumentValue
from yet_another_verb.nomlex.adaptation.argument.arg_renaming import rename_argument
from yet_another_verb.nomlex.adaptation.modifications import get_default_preps_by_arg, \
	get_compatible_args, get_as_arg_type
from yet_another_verb.nomlex.adaptation.subcat.subcat_simplification import add_args_requirement


def enrich_preps_from_nom_subc(subcat: dict, subcat_type: SubcatType, lexicon_type: LexiconType):
	preps = []

	# Verbal arguments are enriched only for prepositional args
	if lexicon_type is LexiconType.VERB and "-P-" not in subcat_type:
		return

	for tag, value in subcat.get("NOM-SUBC", {}).items():
		preps += value.get("PVAL", [])

	if len(preps) == 0:
		return

	# Only sbar and ing args can be enriched
	for complement_type in [ArgumentType.ING, ArgumentType.SBAR]:
		if complement_type not in subcat[SubcatProperty.REQUIRED]:
			continue

		if complement_type not in subcat.keys():
			subcat[complement_type] = preps


def enrich_preps_from_defaults(subcat: dict, subcat_type: SubcatType, lexicon_type: LexiconType):
	default_preps_per_arg = get_default_preps_by_arg(subcat_type, lexicon_type)

	for complement_type, default_preps in default_preps_per_arg.items():
		if complement_type not in subcat.keys() and len(default_preps) > 0:
			subcat[complement_type] = default_preps


def enrich_preps(subcat: dict, subcat_type: SubcatType, lexicon_type: LexiconType):
	enrich_preps_from_nom_subc(subcat, subcat_type, lexicon_type)
	enrich_preps_from_defaults(subcat, subcat_type, lexicon_type)


def use_nom_type(entry: dict, subcat: dict, lexicon_type: LexiconType):
	# Get the type of the nominalization
	nom_type_info = entry[EntryProperty.NOM_TYPE]
	type_of_nom = nom_type_info[NomTypeProperty.TYPE]

	# Find the most compatible argument type for that nom type
	# REMEMBER that the compatible args are ordered by preference
	compatible_args_types = get_compatible_args(type_of_nom)
	new_arg_type = get_as_arg_type(type_of_nom)

	if new_arg_type is None:
		return

	potential_arg_types = subcat[SubcatProperty.REQUIRED] + subcat[SubcatProperty.OPTIONAL]
	existing_arg_types = [t for t in compatible_args_types if t in potential_arg_types]

	if len(existing_arg_types) == 0:
		if lexicon_type == LexiconType.NOUN:
			add_args_requirement([new_arg_type], subcat, force=True)
			subcat[new_arg_type] = [ArgumentValue.NOM]
		return

	existing_arg_type = existing_arg_types[0]

	if lexicon_type == LexiconType.VERB:
		arg_values = list(set(subcat.get(existing_arg_type, []) + nom_type_info[NomTypeProperty.PVAL]))
	else:
		arg_values = [ArgumentValue.NOM]
		add_args_requirement([existing_arg_type], subcat, force=True)  # NOM argument must be required

	# Rename only existing arguments
	subcat[existing_arg_type] = arg_values
	rename_argument(existing_arg_type, new_arg_type, subcat)

	# If PP1 was renamed, then replace PP2 with PP
	if existing_arg_type == ArgumentType.PP1 and new_arg_type != ArgumentType.PP1:
		rename_argument(ArgumentType.PP2, ArgumentType.PP, subcat)


def enrich_arguments(entry: dict, subcat: dict, subcat_type: SubcatType, lexicon_type: LexiconType):
	enrich_preps(subcat, subcat_type, lexicon_type)
	use_nom_type(entry, subcat, lexicon_type)

	for arg_type in set(subcat[SubcatProperty.REQUIRED] + subcat[SubcatProperty.OPTIONAL]):
		if arg_type not in subcat:
			subcat[arg_type] = []
