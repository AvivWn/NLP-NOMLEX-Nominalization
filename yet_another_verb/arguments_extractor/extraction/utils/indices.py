from typing import Tuple, Iterable

IndexRange = Tuple[int, int]


def get_indices_as_range(indices: Iterable[int]) -> IndexRange:
	return min(indices), max(indices)
