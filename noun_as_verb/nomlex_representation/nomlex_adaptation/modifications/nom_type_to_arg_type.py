from typing import List, Union

from noun_as_verb.nomlex_representation.lexicon_constants import NomType, ArgumentType

# The name of the argument that correspond to the nominalization
NOM_TYPE_AS_ARG_TYPE = {
	NomType.SUBJ: ArgumentType.SUBJ,
	NomType.SUBJ_PART: ArgumentType.SUBJ,
	NomType.OBJ: ArgumentType.OBJ,
	NomType.OBJ_PART: ArgumentType.OBJ,
	NomType.IND_OBJ: ArgumentType.IND_OBJ,
	NomType.IND_OBJ_PART: ArgumentType.IND_OBJ
}

# The compatible args are ordered by preferance
# starting with the most compatible
NOM_TYPE_COMPATIBLE_ARGS = {
	NomType.SUBJ: [ArgumentType.SUBJ],
	NomType.SUBJ_PART: [ArgumentType.SUBJ],
	NomType.OBJ: [ArgumentType.OBJ],
	NomType.OBJ_PART: [ArgumentType.OBJ],

	# Sometimes the indirect object actually refer to a PP argument with "to\for" values
	NomType.IND_OBJ: [ArgumentType.IND_OBJ, ArgumentType.PP1, ArgumentType.PP2, ArgumentType.PP],
	NomType.IND_OBJ_PART: [ArgumentType.IND_OBJ, ArgumentType.PP1, ArgumentType.PP2, ArgumentType.PP],

	# Ignored because they are rare and they don't refer to a specific complement
	# NomType.P_OBJ: [ArgumentType.PP],
	# NomType.INSTRUMENT: [ArgumentType.INSTRUMENT]
}


def get_as_arg_type(nom_type: NomType) -> Union[ArgumentType, None]:
	return NOM_TYPE_AS_ARG_TYPE.get(nom_type, None)


def get_compatible_args(nom_type: NomType) -> List[ArgumentType]:
	return NOM_TYPE_COMPATIBLE_ARGS.get(nom_type, [])
