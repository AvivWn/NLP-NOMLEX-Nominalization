from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class CompressedEncoding:
	bytes_data: bytes
	encoding_framework: Optional[str] = field(default=None)
	encoder_name: Optional[str] = field(default=None)

	encoding: Optional[torch.Tensor] = field(default=None)
