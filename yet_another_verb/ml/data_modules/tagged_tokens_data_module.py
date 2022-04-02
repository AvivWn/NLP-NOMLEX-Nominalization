from typing import Optional, List

import pandas as pd
from transformers import BatchEncoding

from yet_another_verb.ml.data_modules.defaults import IGNORED_LABEL_ID
from yet_another_verb.ml.data_modules.pretrained_data_module import PretrainedDataModule


class TaggedTokensDataModule(PretrainedDataModule):
	def __init__(
			self, labels_column: str, main_inputs_column: str, secondary_inputs_column: Optional[str] = None,
			*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self.save_hyperparameters()

	def _align_and_encode_labels(self, examples_labels: List[List[str]], encoded_data: BatchEncoding) -> List[List[int]]:
		encoded_labels = []
		for i, tokens_tags in enumerate(examples_labels):
			original_token_idxs = encoded_data.word_ids(batch_index=i)
			previous_token_idx = None
			label_ids = []
			for token_idx in original_token_idxs:
				if token_idx is not None and token_idx != previous_token_idx:
					label_ids.append(self.tagset[tokens_tags[token_idx]])
				else:
					label_ids.append(IGNORED_LABEL_ID)

				previous_token_idx = token_idx
			encoded_labels.append(label_ids)

		return encoded_labels

	def _encode_data(self, df: pd.DataFrame) -> tuple:
		if self.hparams.secondary_inputs_column is None:
			data = df[self.hparams.main_inputs_column]
		else:
			data = zip(df[self.hparams.main_inputs_column].tolist(), df[self.hparams.secondary_inputs_column].tolist())
			data = map(lambda x: (x[0].split(), [x[1]]), list(data))

		data = list(data)
		encoded_data = self.tokenizer.batch_encode_plus(
			data,
			padding=True,
			return_attention_mask=True,
			truncation=True,
			is_split_into_words=True)
		encoded_labels = self._align_and_encode_labels(df[self.hparams.labels_column].str.split(' ').tolist(), encoded_data)
		return encoded_data["input_ids"], encoded_data["attention_mask"], encoded_labels
