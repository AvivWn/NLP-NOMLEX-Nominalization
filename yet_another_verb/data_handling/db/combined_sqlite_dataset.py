import os
import sqlite3
from shutil import copyfile
from typing import List
from os import listdir
from os.path import isdir, join, dirname
from tempfile import NamedTemporaryFile

from yet_another_verb.data_handling.dataset_creator import DatasetCreator
from yet_another_verb.data_handling.file.file_extensions import DB_EXTENSION
from yet_another_verb.utils.debug_utils import timeit
from yet_another_verb.utils.print_utils import print_if_verbose

MAX_ATTACH_DB = 9
os.environ["SQLITE_TEMPDIR"] = "/home/nlp/avivwn/tmpdir"


class CombinedSQLitesDatasetCreator(DatasetCreator):
	def __init__(self, in_dataset_path: str, **kwargs):
		super().__init__(None)
		self.in_dataset_path = in_dataset_path

	def _get_db_paths(self, file_path: str):
		in_dataset_paths = [file_path]
		if isdir(file_path):
			in_dataset_paths = [join(file_path, file_name) for file_name in listdir(file_path)]

		in_dataset_paths = [file_path for file_path in in_dataset_paths if file_path.endswith(f".{DB_EXTENSION}")]
		return in_dataset_paths if self.dataset_size is None else in_dataset_paths[:self.dataset_size]

	@staticmethod
	def _store_combined_content(db_paths: List[str], combined_db_path: str):
		with sqlite3.connect(combined_db_path) as conn:
			cur = conn.cursor()

			conn.execute("PRAGMA synchronous = OFF;")
			conn.execute("PRAGMA journal_mode = OFF;")
			# conn.execute("PRAGMA temp_stroe_directory = './tmp';")
			conn.commit()

			attaches = []
			for i, db_path in enumerate(db_paths):
				attached_name = f"attached_{i}"
				cur.execute(f"ATTACH '{db_path}' AS {attached_name};")
				attaches.append(attached_name)

			cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
			all_tbls = cur.fetchall()

			for tbl in all_tbls:
				tbl = tbl[0]

				print_if_verbose(f"Populating {tbl}")

				# Delete table content
				timeit(cur.execute)(f"DELETE FROM {tbl};")
				conn.commit()

				# Populate with unioned values
				union_attaches_query = ' UNION '.join(['SELECT * FROM ' + att + '.' + tbl for att in attaches])
				timeit(cur.execute)(f"INSERT INTO {tbl} {union_attaches_query};")
				conn.commit()

				cur.execute(f"SELECT * FROM {tbl}")
				print_if_verbose(f"{tbl}:", len(cur.fetchall()))

			for att in attaches:
				cur.execute(f"DETACH {att};")

			print_if_verbose("VACUUM")
			timeit(cur.execute)(f"VACUUM;")
			conn.commit()

			cur.close()

	def _batch_store_combine_contents(self, db_paths: List[str], out_db_path: str):
		while len([path for path in db_paths if path != out_db_path]) > 0:
			temp_file = NamedTemporaryFile(dir=dirname(out_db_path))

			# Combine all to tmp
			copyfile(db_paths[0], temp_file.name)
			self._store_combined_content(db_paths[:MAX_ATTACH_DB], temp_file.name)

			# Replace tmp with output
			copyfile(temp_file.name, out_db_path)
			db_paths = [out_db_path] + db_paths[MAX_ATTACH_DB:]

			temp_file.close()

	def append_dataset(self, out_dataset_path: str):
		db_paths = self._get_db_paths(self.in_dataset_path)

		temp_file = NamedTemporaryFile(dir=dirname(out_dataset_path))
		copyfile(out_dataset_path, temp_file.name)
		db_paths.append(temp_file.name)

		try:
			self._batch_store_combine_contents(db_paths, out_dataset_path)
		except Exception as e:
			# Return the old content of output on failure
			copyfile(temp_file.name, out_dataset_path)
			raise e

		temp_file.close()

	def create_dataset(self, out_dataset_path: str):
		db_paths = self._get_db_paths(self.in_dataset_path)
		self._batch_store_combine_contents(db_paths, out_dataset_path)
