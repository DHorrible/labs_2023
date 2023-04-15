import random
import numpy as np

from multiprocess.managers import SyncManager
from typing import Any, Dict, List, Tuple
from deap import base, algorithms
from deap import creator
from deap import tools

from matrix import Matrix
from place import Place2D

# Shared sources
__g_mtx: Matrix = None
__g_weights_mtx: np.ndarray = None
def __pool_init(mtx: Matrix, weights_mtx: np.ndarray):
    global __g_mtx, __g_weights_mtx
    __g_mtx = mtx
    __g_weights_mtx = weights_mtx

class EvoAccommodationEvalator(object):
    def __init__(self, mtx: Matrix, weights_mtx: np.ndarray) -> None:
        self._mtx = mtx
        self._weights_mtx = weights_mtx

    def __call__(self, ind: List[int]) -> Tuple[float]:
        _n2i = {x: i for i, x in enumerate(ind)}

        ret = 0
        for i, node in enumerate(ind):
            for cross_node in self._mtx.crosses(node):
                ret += self._mtx.mtx[node, cross_node] * self._weights_mtx[i, _n2i[cross_node]]
        return ret//2,

    def __getstate__(self) -> Dict[str, Any]:
        return {}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._mtx = __g_mtx
        self._weights_mtx = __g_weights_mtx

class EvoAccommodation(object):
    POPULATION_SIZE = 500
    P_CROSSOVER = 0.9
    P_MUTATION = 0.1
    MAX_GENERATIONS = 30
    HALL_OF_FAME_SIZE = 3
    RANDOM_SEED = 42
    WEIGHTS = (-1.,)

    # multiprocess
    PROCESSES = 2

    def __init__(self, mtx: Matrix, manager: SyncManager=None) -> None:
        random.seed(self.RANDOM_SEED)

        self._mtx = mtx
        self._chromo_len = mtx.n

        self._hof = tools.HallOfFame(self.HALL_OF_FAME_SIZE)

        creator.create('FitnessMin', base.Fitness, weights=self.WEIGHTS)
        creator.create('Individual', list, fitness=creator.FitnessMin)

        self._toolbox = base.Toolbox()
        self._toolbox.register(
            'random_order',
            random.sample, range(self._chromo_len), self._chromo_len,
        )
        self._toolbox.register(
            'individual_creator',
            tools.initRepeat, creator.Individual, self._toolbox.random_order, self._chromo_len,
        )
        self._toolbox.register(
            'population_creator',
            tools.initRepeat, list, self._toolbox.individual_creator,
        )

        # TODO 
        self._evalator = EvoAccommodationEvalator(self._mtx, None)
        self._toolbox.register('evaluate', self._evalator)
        self._toolbox.register('select', tools.selTournament, tournsize=3)
        self._toolbox.register('mate', tools.cxOrdered)
        self._toolbox.register('mutate', tools.mutShuffleIndexes, indpb=1.0/self._chromo_len)

        if manager is not None:
            self._toolbox.register(
                'map',
                manager.Pool(
                    processes=self.PROCESSES,
                    initializer=__pool_init,
                    # TODO 
                    initargs=(self._mtx, None),
                ),
            )

        self._stats = tools.Statistics(lambda ind: np.dot(ind.fitness.values, ind.fitness.weights))
        self._stats.register('min', np.min)
        self._stats.register('avg', np.mean)

    def do(self) -> Place2D:
        population = self._toolbox.population_creator(n=self.POPULATION_SIZE)
        population, _ = algorithms.eaSimple(population,
            toolbox=self._toolbox,
            cxpb=self.P_CROSSOVER,
            mutpb=self.P_MUTATION,
            ngen=self.MAX_GENERATIONS,
            halloffame=self._hof,
            stats=self._stats,
            verbose=True,
        )

        ret = self._pop2place(population)
        nodes_seq = ret.to_nodes()

        print(f'Best ind: [ {", ".join(nodes_seq)} ]')
        print(f'Best fitness: {self._evalator(nodes_seq)[0]}')

        return ret

    def _pop2place(self, population: List[List[int]]) -> Place2D:
        population.sort(key=lambda ind: np.dot(ind.fitness.values, ind.fitness.weights))
        best_ind = population[0]
        ret = Place2D(self._mtx)
        for i, x in enumerate(best_ind):
            ret.update(node=x, pos=i)
        return ret
