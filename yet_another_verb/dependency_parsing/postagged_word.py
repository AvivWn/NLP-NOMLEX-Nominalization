from dataclasses import dataclass
from typing import Union

from dataclasses_json import dataclass_json

from yet_another_verb.dependency_parsing import POSTag


@dataclass_json
@dataclass
class POSTaggedWord:
	word: str
	postag: Union[POSTag, str]

	def __hash__(self):
		return hash((self.word, self.postag))
