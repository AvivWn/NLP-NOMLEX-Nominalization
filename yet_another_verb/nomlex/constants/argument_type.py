from enum import Enum


class ArgumentType(str, Enum):
	SUBJ = "SUBJECT"
	OBJ = "OBJECT"
	IND_OBJ = "IND-OBJ"
	NP = "NP"
	PP = "PP"
	PP1 = "PP1"
	PP2 = "PP2"
	MODIFIER = "MODIFIER"
	ING = "ING"
	TO_INF = "TO-INF"
	SBAR = "SBAR"
	PART = "PARTICLE"

	@staticmethod
	def is_np_arg(arg_type: 'ArgumentType'):
		return arg_type in [ArgumentType.OBJ, ArgumentType.SUBJ, ArgumentType.IND_OBJ, ArgumentType.NP]

	@staticmethod
	def is_pp_arg(arg_type: 'ArgumentType'):
		return arg_type in [ArgumentType.PP, ArgumentType.PP1, ArgumentType.PP2]


ARGUMENT_TYPES = [t for t in ArgumentType]
