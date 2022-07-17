from enum import Enum
from typing import List


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
		return arg_type in NP_ARG_TYPES

	@staticmethod
	def is_pp_arg(arg_type: 'ArgumentType'):
		return arg_type in PP_ARG_TYPES


ARGUMENT_TYPES = [t for t in ArgumentType]
NP_ARG_TYPES = [ArgumentType.OBJ, ArgumentType.SUBJ, ArgumentType.IND_OBJ, ArgumentType.NP]
PP_ARG_TYPES = [ArgumentType.PP, ArgumentType.PP1, ArgumentType.PP2]

ArgumentTypes = List[ArgumentType]
