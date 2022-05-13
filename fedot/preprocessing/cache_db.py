import pickle
import sqlite3
import uuid

from contextlib import closing
from pathlib import Path
from typing import Optional

from fedot.core.utils import default_fedot_data_dir
from fedot.preprocessing.preprocessing import DataPreprocessor


class PreprocessingCacheDB:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or Path(default_fedot_data_dir(), f'prp_{str(uuid.uuid4())}')
        self._db_suffix = '.preprocessing_db'
        self.db_path = Path(self.db_path).with_suffix(self._db_suffix)

        self._del_prev_temps()

        self._preproc_table = 'preprocessors'
        self._init_db()

    def get_preprocessor(self, uid: str) -> Optional[DataPreprocessor]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute(f'SELECT preprocessor FROM {self._preproc_table} WHERE id = ?;', [uid])
                matched = cur.fetchone()
                if matched is not None:
                    matched = pickle.loads(matched[0])
                return matched

    def add_preprocessor(self, uid: str, value: DataPreprocessor):
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                db_pickled = sqlite3.Binary(pickle.dumps(value, pickle.HIGHEST_PROTOCOL))
                cur.execute(f'INSERT OR IGNORE INTO {self._preproc_table} VALUES (?, ?);', [uid, db_pickled])

    def reset(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute(f'DELETE FROM {self._preproc_table};')

    def _init_db(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute((
                    f'CREATE TABLE IF NOT EXISTS {self._preproc_table} ('
                    'id TEXT PRIMARY KEY,'  # noqa better viewed like that
                    'preprocessor BLOB'  # noqa
                    ');'
                ))

    def _del_prev_temps(self):
        for file in self.db_path.parent.glob(f'prp_*{self._db_suffix}'):
            file.unlink()
