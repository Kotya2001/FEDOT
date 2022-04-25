from copy import deepcopy
from itertools import zip_longest
from typing import Any, List, Optional, Tuple, Union, Callable

import numpy as np
from deap import tools
from tqdm import tqdm

from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.gp_operators import (
    clean_operators_history,
    duplicates_filtration,
    num_of_parents_in_crossover
)
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.iterator import SequenceIterator, fibonacci_sequence
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum, inheritance
from fedot.core.optimisers.gp_comp.generation_keeper import best_individual
from fedot.core.optimisers.gp_comp.operators.regularization import regularized_population
from fedot.core.optimisers.gp_comp.operators.selection import selection
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.optimisers.utils.population_utils import is_equal_archive
from fedot.core.repository.quality_metrics_repository import ComplexityMetricsEnum, MetricsEnum, MetricsRepository

DEFAULT_MAX_POP_SIZE = 55


class EvoGraphParameterFreeOptimiser(EvoGraphOptimiser):
    """
    Implementation of the parameter-free adaptive evolutionary optimiser
    (population size and genetic operators rates is changing over time).
    For details, see https://ieeexplore.ieee.org/document/9504773
    """

    def __init__(self, initial_graph, requirements, graph_generation_params, metrics: List[MetricsEnum],
                 parameters: Optional[GPGraphOptimiserParameters] = None,
                 max_population_size: int = DEFAULT_MAX_POP_SIZE,
                 sequence_function=fibonacci_sequence, log: Log = None):
        super().__init__(initial_graph, requirements, graph_generation_params, metrics, parameters, log)

        if self.parameters.genetic_scheme_type is not GeneticSchemeTypesEnum.parameter_free:
            self.log.warn(f'Invalid genetic scheme type was changed to parameter-free. Continue.')
            self.parameters.genetic_scheme_type = GeneticSchemeTypesEnum.parameter_free

        self.sequence_function = sequence_function
        self.max_pop_size = max_population_size
        self.iterator = SequenceIterator(sequence_func=self.sequence_function, min_sequence_value=1,
                                         max_sequence_value=self.max_pop_size,
                                         start_value=self.requirements.pop_size)

        self.requirements.pop_size = self.iterator.next()

        self.stopping_after_n_generation = parameters.stopping_after_n_generation

    def optimise(self, objective_function, offspring_rate: float = 0.5,
                 on_next_iteration_callback: Optional[Callable] = None,
                 intermediate_metrics_function: Optional[Callable] = None,
                 show_progress: bool = True) -> Union[OptGraph, List[OptGraph]]:

        self.evaluator.objective_function = objective_function  # TODO: move into init!

        if on_next_iteration_callback is None:
            on_next_iteration_callback = self.default_on_next_iteration_callback

        # TODO: leave this eval at the beginning of loop
        num_of_new_individuals = self.offspring_size(offspring_rate)
        self.log.info(f'pop size: {self.requirements.pop_size}, num of new inds: {num_of_new_individuals}')

        with self.timer as t:
            pbar = tqdm(total=self.requirements.num_of_generations,
                        desc='Generations', unit='gen', initial=1,
                        disable=self.log.verbosity_level == -1) if show_progress else None

            self._init_population()
            self.population = self.evaluator(self.population)
            self.generations.append(self.population)

            on_next_iteration_callback(self.population, self.generations.best_individuals)
            self.log_info_about_best()

            while not self.stop_optimisation():
                self.log.info(f'Generation num: {self.generations.generation_num}')
                self.log.info(f'max_depth: {self.max_depth}, no improvements: {self.generations.stagnation_length}')

                # TODO: subst to mutation params
                if self.parameters.with_auto_depth_configuration and self.generations.generation_num > 0:
                    self.max_depth_recount()

                self.max_std = self.update_max_std()

                individuals_to_select = \
                    regularized_population(reg_type=self.parameters.regularization_type,
                                           population=self.population,
                                           objective_function=objective_function,
                                           graph_generation_params=self.graph_generation_params,
                                           timer=t)

                if self.parameters.multi_objective:
                    # TODO: feels unneeded, ParetoFront does it anyway
                    filtered_archive_items = duplicates_filtration(self.generations.best_individuals, individuals_to_select)
                    individuals_to_select = deepcopy(individuals_to_select) + filtered_archive_items

                # TODO: collapse this selection & reprodue for 1 and for many
                if num_of_new_individuals == 1 and len(self.population) == 1:
                    new_population = list(self.reproduce(self.population[0]))
                    new_population = self.evaluator(new_population)
                else:
                    num_of_parents = num_of_parents_in_crossover(num_of_new_individuals)

                    selected_individuals = selection(types=self.parameters.selection_types,
                                                     population=individuals_to_select,
                                                     pop_size=num_of_parents,
                                                     params=self.graph_generation_params)

                    new_population = []

                    for ind_1, ind_2 in zip_longest(selected_individuals[::2], selected_individuals[1::2]):
                        new_population += self.reproduce(ind_1, ind_2)

                    new_population = self.evaluator(new_population)

                # TODO: make internally used iterator allow initial run (at loop beginning)
                self.requirements.pop_size = self.next_population_size()
                # TODO: move to loop beginning
                num_of_new_individuals = self.offspring_size(offspring_rate)  # leaves iterator unchanged
                self.log.info(f'pop size: {self.requirements.pop_size}, num of new inds: {num_of_new_individuals}')

                self.population = inheritance(self.parameters.genetic_scheme_type, self.parameters.selection_types,
                                              self.population,
                                              new_population, self.num_of_inds_in_next_pop,
                                              graph_params=self.graph_generation_params)

                # Add best individuals from the previous generation
                if not self.parameters.multi_objective and self.with_elitism:
                    self.population.extend(self.generations.best_individuals)
                # Then update generation
                self.generations.append(self.population)

                # TODO: move into dynamic mutation operator
                if not self.generations.last_improved:
                    self.operators_prob_update()

                on_next_iteration_callback(self.population, self.generations.best_individuals)
                self.log.info(f'spent time: {round(t.minutes_from_start, 1)} min')
                self.log_info_about_best()

                clean_operators_history(self.population)

                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.close()

            best = self.generations.best_individuals
            self.log.info('Result:')
            self.log_info_about_best()

        return self.to_outputs(best)

    @property
    def with_elitism(self) -> bool:
        if self.parameters.multi_objective:
            return False
        else:
            return self.requirements.pop_size >= 7

    @property
    def current_std(self):
        return np.std([ind.fitness.value for ind in self.population])

    def update_max_std(self):
        if self.generations.generation_num <= 1:
            std_max = self.current_std
            if len(self.population) == 1:
                self.requirements.mutation_prob = 1
                self.requirements.crossover_prob = 0
            else:
                self.requirements.mutation_prob = 0.5
                self.requirements.crossover_prob = 0.5
        else:
            if self.max_std < self.current_std:
                std_max = self.current_std
            else:
                std_max = self.max_std
        return std_max

    def next_population_size(self) -> int:
        fitness_improved = self.generations.quality_improved
        complexity_decreased = self.generations.complexity_improved
        progress_in_both_goals = fitness_improved and complexity_decreased
        no_progress = not fitness_improved and not complexity_decreased

        next_population_size = len(self.population)
        if progress_in_both_goals and len(self.population) > 2:
            if self.iterator.has_prev():
                next_population_size = self.iterator.prev()
        elif no_progress:
            if self.iterator.has_next():
                next_population_size = self.iterator.next()

        return next_population_size

    def operators_prob_update(self):
        std = float(self.current_std)
        max_std = float(self.max_std)

        mutation_prob = 1 - (std / max_std) if max_std > 0 and std != max_std else 0.5
        crossover_prob = 1 - mutation_prob

        self.requirements.mutation_prob = mutation_prob
        self.requirements.crossover_prob = crossover_prob

    def offspring_size(self, offspring_rate: float = None) -> int:
        if self.iterator.has_prev():
            num_of_new_individuals = self.iterator.prev()
            self.iterator.next()
        else:
            num_of_new_individuals = 1
        return num_of_new_individuals
