from enum import Enum


class NomTypeProperty(str, Enum):
	TYPE = "TYPE"
	PART = "PART"
	PVAL = "PVAL"
	ADVAL = "ADVAL"


NOM_TYPE_PROPERTIES = {p for p in NomTypeProperty}
