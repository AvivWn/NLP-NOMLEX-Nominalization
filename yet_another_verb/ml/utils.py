from typing import List, Dict

from toolz import itemmap

Tagset = Dict[str, int]


def labels_to_tagset(labels: List[str]) -> Tagset:
	return itemmap(reversed, dict(enumerate(sorted(labels))))
