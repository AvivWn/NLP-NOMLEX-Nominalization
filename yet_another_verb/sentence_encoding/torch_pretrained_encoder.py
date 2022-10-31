import torch
from transformers import AutoTokenizer, AutoModel

from yet_another_verb.sentence_encoding.encoder import Encoder


class TorchPretrainedEncoder(Encoder):
	def __init__(self, encoder_name: str, device: str = None, **kwargs):
		self.device = device
		if self.device is None:
			self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.encoder_name = encoder_name
		self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
		self.model = AutoModel.from_pretrained(encoder_name).to(self.device)
		self.model.eval()

	@property
	def name(self) -> str:
		return self.encoder_name

	def encode(self, sentence: str) -> torch.Tensor:
		tokenized = self.tokenizer(sentence.split(), return_tensors="pt", is_split_into_words=True, add_special_tokens=True)
		tokenized = tokenized.to(self.device)

		with torch.no_grad():
			return self.model(**tokenized)[0][0].cpu()
