from typing import Union

import torch

from yet_another_verb.data_handling.bytes.compressed.compressed_encoding import CompressedEncoding

Encoding = Union[torch.Tensor, CompressedEncoding]
