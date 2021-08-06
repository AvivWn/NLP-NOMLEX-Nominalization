from enum import Enum


class UPOSTag(str, Enum):
	VERB = "VERB"
	PART = "PART"
	ADP = "ADP"
	ADJ = "ADJ"
	ADV = "ADV"
	NOUN = "NOUN"
	PROPN = "PROPN"
	PRON = "PRON"
	DET = "DET"
	AUX = "AUX"
	PUNCT = "PUNCT"


U_POSTAGS = {t for t in UPOSTag}
