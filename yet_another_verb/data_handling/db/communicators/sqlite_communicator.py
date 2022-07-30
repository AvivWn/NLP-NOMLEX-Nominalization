from os.path import exists

from pony.orm import Database

from yet_another_verb.data_handling.db.communicators.communicator import DBCommunicator


class SQLiteCommunicator(DBCommunicator):
	def __init__(self, db: Database, db_path: str, create_db=False):
		self.db = db
		self.db_path = db_path
		self.create_db = create_db

		self.connect()

	def connect(self):
		if not self.create_db and not exists(self.db_path):
			raise FileNotFoundError()

		self.db.bind(provider='sqlite', filename=self.db_path, create_db=self.create_db)

	def disconnect(self):
		self.db.disconnect()

	def generate_mapping(self):
		if self.db.schema is None:
			self.db.generate_mapping(create_tables=self.create_db)

	def commit(self):
		self.db.commit()

	def execute(self, sql: str):
		self.db.execute(sql)

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, exc_traceback):
		self.disconnect()
		self.db.provider = None
		self.db.schema = None
