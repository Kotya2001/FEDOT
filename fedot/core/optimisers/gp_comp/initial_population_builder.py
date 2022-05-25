from typing import Callable, Optional, Sequence

import numpy as np

from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.operator import Operator, PopulationT
from fedot.core.optimisers.graph import OptGraph
from fedot.core.pipelines.validation import GraphValidator

GraphSampler = Callable[[], OptGraph]


class InitialPopulationBuilder:
    """Generates initial population using two approaches.
    One is with initial graphs that are augmented and randomized with mutation operator.
    Another is just sampling and validating graphs from provided graph sampler."""

    def __init__(self, validator: GraphValidator):
        self.validator = validator
        self.mutation_operator: Callable[[Individual], Individual] = lambda ind: ind
        self.graph_sampler: Optional[GraphSampler] = None
        self.initial_graphs: Sequence[OptGraph] = ()

    def with_mutated_inds(self, mutation_operator: Operator[Individual]):
        self.mutation_operator = mutation_operator
        return self

    def with_initial_individuals(self, initial_graphs: Sequence[OptGraph]):
        if initial_graphs:
            self.initial_graphs = initial_graphs
            self.graph_sampler = lambda: np.random.choice(self.initial_graphs)
        return self

    def with_custom_sampler(self, sampler: GraphSampler):
        self.graph_sampler = sampler
        return self

    def build(self, pop_size: int) -> PopulationT:
        if self.graph_sampler is None:
            raise ValueError("Provide sampler of initial individuals")

        population = []
        population.extend(Individual(graph) for graph in self.initial_graphs)
        while len(population) < pop_size:
            new_ind = Individual(self.graph_sampler())
            new_ind = self.mutation_operator(new_ind)
            if new_ind not in population and self.validator(new_ind.graph):
                population.append(new_ind)

        return population
