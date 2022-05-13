from typing import TYPE_CHECKING, Optional

from fedot.core.log import Log, SingletonMeta, default_log
from fedot.preprocessing.cache_db import PreprocessingCacheDB

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline


class PreprocessingCache(metaclass=SingletonMeta):
    def __init__(self, log: Optional[Log] = None, db_path: Optional[str] = None):
        self.log = log or default_log(__name__)
        self._db = PreprocessingCacheDB(db_path)

    def try_find_preprocessor(self, pipeline: 'Pipeline'):
        try:
            descriptive_id = pipeline.root_node.descriptive_id
            matched = self._db.get_preprocessor(descriptive_id)
            if matched is not None:
                return matched
        except Exception as exc:
            self.log.error(f'Preprocessor search error: {exc}')
        return pipeline.preprocessor

    def add_preprocessor(self, pipeline: 'Pipeline'):
        self._db.add_preprocessor(pipeline.root_node.descriptive_id, pipeline.preprocessor)

    def reset(self):
        self._db.reset()
