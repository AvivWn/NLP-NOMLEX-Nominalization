from enum import Enum


class NomType(str, Enum):
	VERB_NOM = "VERB-NOM"
	VERB_PART = "VERB-PART"
	SUBJ = "SUBJECT"
	SUBJ_PART = "SUBJECT-PART"
	OBJ = "OBJECT"
	OBJ_PART = "OBJECT-PART"
	IND_OBJ = "IND-OBJ"
	IND_OBJ_PART = "IND-OBJ-PART"
	P_OBJ = "P-OBJ"
	P_OBJ_PART = "P-OBJ-PART"


NOM_TYPES = {t for t in NomType}
