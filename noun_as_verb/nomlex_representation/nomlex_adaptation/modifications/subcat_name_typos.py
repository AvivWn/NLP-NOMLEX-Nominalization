SUBCAT_NAME_TYPOS = {
	"NOM-INSTRANS": "NOM-INTRANS",
	"INTRANS": "NOM-INTRANS"
}


def get_correct_subcat_type(subcat_type: str) -> str:
	if subcat_type in SUBCAT_NAME_TYPOS:
		return SUBCAT_NAME_TYPOS[subcat_type]

	return subcat_type
