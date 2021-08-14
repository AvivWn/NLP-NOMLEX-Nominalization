from enum import Enum


class EntryType(str, Enum):
	NOM = "NOM"


ENTRY_TYPES = {t for t in EntryType}
