from enum import Enum


class SubcatProperty(str, Enum):
	ARGUMENTS = "ARGUMENTS"
	REQUIRED = "REQUIRED"
	OPTIONAL = "OPTIONAL"
	NOT = "NOT"

	# Only in original
	N_N_MOD_NO_OTHER_OBJ = "N-N-MOD-NO-OTHER-OBJ"
	DET_POSS_NO_OTHER_OBJ = "DET-POSS-NO-OTHER-OBJ"
	ALTERNATES = "ALTERNATES"


SUBCAT_PROPERTIES = {p for p in SubcatProperty}
