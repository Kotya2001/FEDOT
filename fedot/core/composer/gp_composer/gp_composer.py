import platform
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from multiprocessing import set_start_method
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, Union

from fedot.core.composer.cache import OperationsCache
from fedot.core.composer.composer import Composer, ComposerRequirements
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log, default_log
from fedot.core.optimisers.gp_comp.operators.mutation import MutationStrengthEnum
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.optimizer import GraphOptimiser
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.validation import common_rules, ts_rules, validate
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.validation.objective import Objective
from fedot.core.validation.objective_eval import DataObjectiveEvaluate, ObjectiveEvaluate, DataSource
from fedot.core.validation.split import tabular_cv_generator, ts_cv_generator
from fedot.remote.remote_evaluator import RemoteEvaluator, init_data_for_remote_execution

sample_split_ratio_for_tasks = {
    TaskTypesEnum.classification: 0.8,
    TaskTypesEnum.regression: 0.8,
    TaskTypesEnum.ts_forecasting: 0.5
}


def set_multiprocess_start_method():
    system = platform.system()
    if system == 'Linux':
        set_start_method("spawn", force=True)


@dataclass
class PipelineComposerRequirements(ComposerRequirements):
    """
    Dataclass is for defining the requirements for composition process of genetic programming composer

    :attribute pop_size: population size
    :attribute num_of_generations: maximal number of evolutionary algorithm generations
    :attribute crossover_prob: crossover probability (the chance that two chromosomes exchange some of their parts)
    :attribute mutation_prob: mutation probability
    :attribute mutation_strength: strength of mutation in tree (using in certain mutation types)
    :attribute start_depth: start value of tree depth
    :attribute validation_blocks: number of validation blocks for time series validation
    :attribute n_jobs: num of n_jobs
    :attribute collect_intermediate_metric: save metrics for intermediate (non-root) nodes in pipeline
    """
    pop_size: Optional[int] = 20
    num_of_generations: Optional[int] = 20
    offspring_rate: Optional[float] = 0.5
    crossover_prob: Optional[float] = 0.8
    mutation_prob: Optional[float] = 0.8
    mutation_strength: MutationStrengthEnum = MutationStrengthEnum.mean
    start_depth: int = None
    validation_blocks: int = None
    n_jobs: int = 1
    collect_intermediate_metric: bool = False


class GPComposer(Composer):
    """
    Genetic programming based composer
    :param optimiser: optimiser generated in ComposerBuilder.
    :param composer_requirements: requirements for composition process.
    :param initial_pipelines: defines the initial state of the population. If None then initial population is random.
    :param logger: optional Log object for logging.
    :param cache: optional cache for Operations.
    """

    def __init__(self, optimiser: GraphOptimiser,
                 composer_requirements: PipelineComposerRequirements,
                 initial_pipelines: Optional[Sequence[Pipeline]] = None,
                 logger: Optional[Log] = None,
                 cache: Optional[OperationsCache] = None):

        super().__init__(optimiser, composer_requirements, initial_pipelines, logger)

        self.optimiser = optimiser
        self.cache: Optional[OperationsCache] = cache

        self.objective_builder = DataObjectiveBuilder(self.optimiser.objective,
                                                      self.composer_requirements.max_pipeline_fit_time,
                                                      self.composer_requirements.cv_folds,
                                                      self.composer_requirements.validation_blocks,
                                                      self.cache, self.log)

    def compose_pipeline(self, data: Union[InputData, MultiModalData]) -> Union[Pipeline, List[Pipeline]]:
        self.optimiser.graph_generation_params.advisor.task = data.task

        # TODO: move this late-init logic to the point before optimiser is constructed
        if data.task.task_type is TaskTypesEnum.ts_forecasting:
            self.optimiser.graph_generation_params.rules_for_constraint = ts_rules + common_rules
        else:
            self.optimiser.graph_generation_params.rules_for_constraint = common_rules

        if self.composer_requirements.max_pipeline_fit_time:
            set_multiprocess_start_method()

        # shuffle data if necessary
        data.shuffle()

        objective_evaluator = self.objective_builder.build(data)
        opt_result = self.optimiser.optimise(objective_evaluator)
        best_pipeline = self._convert_opt_results_to_pipeline(opt_result)
        self.log.info('GP composition finished')
        return best_pipeline

    def _convert_opt_results_to_pipeline(self, opt_result: Union[OptGraph, List[OptGraph]]) -> Pipeline:
        return [self.optimiser.graph_generation_params.adapter.restore(graph)
                for graph in opt_result] if isinstance(opt_result, list) \
            else self.optimiser.graph_generation_params.adapter.restore(opt_result)

    @staticmethod
    def tune_pipeline(pipeline: Pipeline, data: InputData, time_limit):
        raise NotImplementedError()

    @property
    def history(self):
        return self.optimiser.history


class DataObjectiveBuilder:

    def __init__(self,
                 objective: Objective,
                 max_pipeline_fit_time: Optional[timedelta] = None,
                 cv_folds: Optional[int] = None,
                 validation_blocks: Optional[int] = None,
                 cache: Optional[OperationsCache] = None,
                 log: Log = None):

        self.objective = objective
        self.max_pipeline_fit_time = max_pipeline_fit_time
        self.cv_folds = cv_folds
        self.validation_blocks = validation_blocks
        self.cache = cache
        self.log = log or default_log(self.__class__.__name__)

    def build(self, data: InputData) -> ObjectiveEvaluate:
        if self.cv_folds is not None:
            data_producer = self._build_kfolds_producer(data)
        else:
            data_producer = self._build_holdout_producer(data)

        objective_evaluate = DataObjectiveEvaluate(objective=self.objective,
                                                   data_producer=data_producer,
                                                   time_constraint=self.max_pipeline_fit_time,
                                                   validation_blocks=self.validation_blocks,
                                                   cache=self.cache, log=self.log)
        return objective_evaluate

    def _build_holdout_producer(self, data: InputData) -> DataSource:
        """Build trivial data producer for hold-out validation
        that always returns same data split. Equivalent to 1-fold validation."""

        self.log.info("Hold out validation for graph composing was applied.")
        split_ratio = sample_split_ratio_for_tasks[data.task.task_type]
        train_data, test_data = train_test_data_setup(data, split_ratio)

        def data_producer(): yield train_data, test_data

        if RemoteEvaluator().use_remote:
            init_data_for_remote_execution(train_data)

        return data_producer

    def _build_kfolds_producer(self, data: InputData) -> DataSource:
        if isinstance(data, MultiModalData):
            raise NotImplementedError('Cross-validation is not supported for multi-modal data')
        if data.task.task_type is TaskTypesEnum.ts_forecasting:
            # Perform time series cross validation
            self.log.info("Time series cross validation for pipeline composing was applied.")
            if self.validation_blocks is None:
                default_validation_blocks = 3
                self.log.info(f'For ts cross validation validation_blocks number was changed ' +
                              f'from None to {default_validation_blocks} blocks')
                # NB: this change isn't propagated back into ComposerRequirements!
                self.validation_blocks = default_validation_blocks
            cv_generator = partial(ts_cv_generator, data,
                                   self.cv_folds,
                                   self.validation_blocks,
                                   self.log)
        else:
            self.log.info("KFolds cross validation for pipeline composing was applied.")
            cv_generator = partial(tabular_cv_generator, data,
                                   self.cv_folds)
        return cv_generator
