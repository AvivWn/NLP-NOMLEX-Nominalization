import random
from typing import List

from yet_another_verb.data_handling.dataset_creator import DatasetCreator
from yet_another_verb.data_handling.file.handlers.txt_file_handler import TXTFileHandler


class ShuffledLinesDatasetCreator(DatasetCreator):
	def __init__(self, in_dataset_path: str, dataset_size=None, **kwargs):
		super().__init__(dataset_size)
		self.in_dataset_path = in_dataset_path

	def _get_shuffled_sentences(self, sentences: List[str]) -> List[str]:
		sentences = [s.strip() for s in sentences][:self.dataset_size]
		random.Random(42).shuffle(sentences)
		return [s + "\n" for s in sentences]

	def create_dataset(self, out_dataset_path):
		in_dataset = TXTFileHandler().load(self.in_dataset_path)
		out_dataset = self._get_shuffled_sentences(in_dataset)
		TXTFileHandler().save(out_dataset_path, out_dataset)
