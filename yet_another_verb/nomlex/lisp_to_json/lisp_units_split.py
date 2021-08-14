from typing import List
import re


def split_lisp_into_units(lisp_text: str) -> List[str]:
	"""
	Splits the given text into basic lisp units
	:param lisp_text: lisp formatted object as string
	:return: a list of lisp units
	"""

	# Spacing up all the opening\closing brackets
	lisp_text = lisp_text.replace("(", " ( ").replace(")", " ) ").replace("\\\"", "")

	# Deal with a phrase of several words (seperated with spaces) as a one word
	lst = lisp_text.split('"')
	for i, item in enumerate(lst):
		if i % 2 == 1:
			lst[i] = re.sub(r"\s+", "_", item)

	return [x.replace("_", " ") for x in '"'.join(lst).split(" ") if x != ""]
