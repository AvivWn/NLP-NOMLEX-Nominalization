from io import BytesIO

import torch

from yet_another_verb.data_handling.bytes.bytes_handler import BytesHandler


class TorchBytesHandler(BytesHandler):
	@staticmethod
	def loads(bytes_data: bytes) -> torch.Tensor:
		buff = BytesIO(bytes_data)
		return torch.load(buff)

	@staticmethod
	def saves(tensor: torch.Tensor) -> bytes:
		buff = BytesIO()
		torch.save(tensor, buff)
		buff.seek(0)
		return buff.read()
