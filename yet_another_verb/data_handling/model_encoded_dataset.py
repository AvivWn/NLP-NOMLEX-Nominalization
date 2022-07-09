from typing import Iterator, List

import torch
from torch import tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.data_handling import PKLFileHandler, ParsedBinFileHandler
from yet_another_verb.data_handling.dataset_creator import DatasetCreator

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


class ModelEncodedDatasetCreator(DatasetCreator):
	def __init__(
			self, in_dataset_path: str,
			model_name: str, device: str,
			dependency_parser: DependencyParser,
			dataset_size=None, **kwargs
	):
		super().__init__(dataset_size)
		self.in_dataset_path = in_dataset_path

		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModel.from_pretrained(model_name).cuda()
		# self.device = device
		self.model.eval()

		self.dependency_parser = dependency_parser

	def _encode_docs(self, docs: Iterator[ParsedText]) -> List[tensor]:
		encodings = []

		for doc in tqdm(docs, leave=False):
			tokenized = self.tokenizer(doc.words, return_tensors="pt", is_split_into_words=True)
			tokenized = tokenized.to("cuda")

			with torch.no_grad():
				encoding = self.model(**tokenized)[0][0].cpu()
				encodings.append(encoding)

			if self.has_reached_size(encodings):
				break

		return encodings

	def create_dataset(self, out_dataset_path):
		parsed_bin = ParsedBinFileHandler(self.dependency_parser).load(self.in_dataset_path)
		in_dataset = parsed_bin.get_parsed_texts()
		out_dataset = self._encode_docs(in_dataset)
		PKLFileHandler.save(out_dataset_path, out_dataset)
