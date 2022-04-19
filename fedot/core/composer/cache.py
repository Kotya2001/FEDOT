import shelve
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, TypeVar, Union, Type

from fedot.core.log import Log, SingletonMeta, default_log
from fedot.core.operations.operation import Operation
from fedot.core.pipelines.node import Node
from fedot.core.utilities.data_structures import ensure_list

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline

from fedot.core.utils import default_fedot_data_dir
from contextlib import nullcontext, contextmanager
from multiprocessing.managers import SyncManager

IOperation = TypeVar('IOperation', bound=Operation)


@dataclass
class CachedState:
    operation: IOperation


class OperationsCache(metaclass=SingletonMeta):
    """
    Stores/loads nodes `fitted_operation` field to increase performance of calculations.

    :param log: optional Log object to record messages
    :param db_path: optional str determining a file name for caching pipelines
    :param clear_exiting: optional bool indicating if it is needed to clean up resources before class can be used
    """

    def __init__(self, log: Optional[Log] = None, db_path: Optional[str] = None, clear_exiting=True):
        self.log = log or default_log(__name__)
        self.db_path = db_path or Path(str(default_fedot_data_dir()), f'tmp_{str(uuid.uuid4())}').as_posix()

        self._rlock = nullcontext()
        effectiveness_keys = ['pipelines_hit', 'nodes_hit', 'pipelines_total', 'nodes_total']
        self._effectiveness = dict.fromkeys(effectiveness_keys, 0)

        if clear_exiting:
            self._clear()

    @contextmanager
    def using_multiprocessing(self, mp_manager: SyncManager):
        self._rlock = mp_manager.RLock()
        self._effectiveness = mp_manager.dict(self._effectiveness)
        yield
        self._rlock = nullcontext()
        self._effectiveness = dict(self._effectiveness)

    @contextmanager
    def using_resources(self):
        self._clear()
        try:
            yield
        finally:
            self._clear()

    @property
    def effectiveness_ratio(self):
        """
        Returns percent of how many pipelines/nodes were loaded instead of computing
        """
        pipelines_hit = self._effectiveness['pipelines_hit']
        pipelines_total = self._effectiveness['pipelines_total']
        nodes_hit = self._effectiveness['nodes_hit']
        nodes_total = self._effectiveness['nodes_total']

        return {
            'pipelines': round(pipelines_hit / pipelines_total, 3) if pipelines_total else 0.,
            'nodes': round(nodes_hit / nodes_total, 3) if nodes_total else 0.
        }

    def reset(self):
        for k in self._effectiveness:
            self._effectiveness[k] = 0
        self._clear()

    def save_nodes(self, nodes: Union[Node, List[Node]], fold_id: Optional[int] = None):
        """
        :param nodes: node/nodes for caching
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        with self._rlock:
            try:
                with shelve.open(self.db_path) as cache:
                    for node in ensure_list(nodes):
                        _save_cache_for_node(cache, node, fold_id)
            except Exception as ex:
                self.log.info(f'Nodes can not be saved: {ex}. Continue')

    def save_pipeline(self, pipeline: 'Pipeline', fold_id: Optional[int] = None):
        """
        :param pipeline: pipeline for caching
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        self.save_nodes(pipeline.nodes, fold_id)

    def try_load_nodes(self, nodes: Union[Node, List[Node]], fold_id: Optional[int] = None) -> bool:
        """
        :param nodes: nodes which fitted state should be loaded from cache
        :param fold_id: optional part of cache item UID
                            (can be used to specify the number of CV fold)
        """
        with self._rlock:
            cache_was_used = False
            try:
                with shelve.open(self.db_path) as cache:
                    for node in ensure_list(nodes):
                        cached_state = _load_cache_for_node(cache, node, fold_id)
                        if cached_state is not None:
                            node.fitted_operation = cached_state.operation
                            cache_was_used = True
                            self._effectiveness['nodes_hit'] += 1
                        else:
                            node.fitted_operation = None
                        self._effectiveness['nodes_total'] += 1
            except Exception as ex:
                self.log.info(f'Cache can not be loaded: {ex}. Continue.')
            finally:
                return cache_was_used

    def try_load_into_pipeline(self, pipeline: 'Pipeline', fold_id: Optional[int] = None) -> bool:
        """
        :param pipeline: pipeline for loading cache into
        :param fold_id: optional part of cache item UID
                            (number of the CV fold)
        """
        with self._rlock:
            hits_before = self._effectiveness['nodes_hit']
            did_load_any = self.try_load_nodes(pipeline.nodes, fold_id)
            hits_after = self._effectiveness['nodes_hit']

            if hits_after - hits_before == len(pipeline.nodes):
                self._effectiveness['pipelines_hit'] += 1
            self._effectiveness['pipelines_total'] += 1

            return did_load_any

    def _clear(self):
        db_path = Path(self.db_path)
        for file in db_path.parent.iterdir():
            if file.stem == db_path.stem:
                if file.is_dir():
                    shutil.rmtree(file)
                else:
                    file.unlink()


def _get_structural_id(node: Node, fold_id: Optional[int] = None) -> str:
    structural_id = node.descriptive_id
    structural_id += f'_{fold_id}' if fold_id is not None else ''
    return structural_id


def _save_cache_for_node(cache_shelf: shelve.Shelf, node: Node, fold_id: Optional[int] = None):
    if node.fitted_operation is not None:
        cached_state = CachedState(node.fitted_operation)
        structural_id = _get_structural_id(node, fold_id)
        cache_shelf[structural_id] = cached_state


def _load_cache_for_node(cache_shelf: shelve.Shelf,
                         node: Node, fold_id: Optional[int] = None) -> Optional[Type[CachedState]]:
    structural_id = _get_structural_id(node, fold_id)
    cached_state = cache_shelf.get(structural_id, None)

    return cached_state
