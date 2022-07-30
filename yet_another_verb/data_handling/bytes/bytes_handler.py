import abc


class BytesHandler(abc.ABC):
	@staticmethod
	@abc.abstractmethod
	def loads(bytes_data: bytes) -> object:
		pass

	@staticmethod
	@abc.abstractmethod
	def saves(data: object) -> bytes:
		pass
