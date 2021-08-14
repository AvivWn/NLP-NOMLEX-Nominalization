from enum import Enum


class ArgumentValue(str, Enum):
	# NP
	N_N_MOD = "N-N-MOD"
	DET_POSS = "DET-POSS"
	NSUBJ = "NSUBJ"
	NSUBJPASS = "NSUBJPASS"
	DOBJ = "DOBJ"
	IOBJ = "IOBJ"
	NP = "NP"

	# MODIFIER
	AJMOD = "AJMOD"
	ADMOD = "ADMOD"

	# TO-INF
	TO_INF = "TO-INF"

	# ING
	ING = "ING"
	POSSING = "POSSING"

	# SBAR
	SBAR = "SBAR"
	THAT_S = "THAT"
	WHAT_S = "WHAT-S"
	WHETHER_S = "WHETHER-S"
	IF_S = "IF-S"
	WHERE_S = "WHERE-S"
	WHEN_S = "WHEN-S"
	HOW_MUCH_S = "HOW-MUCH-S"
	HOW_MANY_S = "HOW-MANY-S"
	HOW_S = "HOW-S"
	HOW_TO_INF = "HOW-TO-INF"
	AS_IF_S = "AS-IF-S"

	# OTHERS
	PART = "PART"
	NOM = "NOM"
	NONE = "NONE"


ARGUMENT_VALUES = [v for v in ArgumentValue]
