import abc


class Factory(abc.ABC):
	@abc.abstractmethod
	def __call__(self):
		pass
