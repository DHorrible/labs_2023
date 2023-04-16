import random
import numpy as np

from multiprocess.managers import SyncManager
from typing import Any, Dict, Iterable, List, Tuple
from deap import base, algorithms
from deap import creator
from deap import tools

from matrix import Matrix
from place import Place2D

# Shared sources
g_mtx: Matrix = None
g_route_mtx: np.ndarray = None
def pool_init(mtx: Matrix, route_mtx: np.ndarray):
    global g_mtx, g_route_mtx
    g_mtx = mtx
    g_route_mtx = route_mtx

class Fitness(base.Fitness):
    def __init__(self, values: Iterable[float]=()):
        super().__init__(values)

    @staticmethod
    def patch_weights(weights: Iterable[float]):
        Fitness.weights = weights

class Individual(list):
    def __init__(self, iterable):
        self.fitness = Fitness()
        super().__init__(iterable)

class EvoAccommodationEvalator(object):
    def __init__(self, mtx: Matrix, route_mtx: np.ndarray) -> None:
        global g_mtx, g_route_mtx
        self._mtx = g_mtx = mtx
        self._route_mtx = g_route_mtx = route_mtx

    def __call__(self, ind: List[int]) -> Tuple[float]:
        # Revers ind array index
        _n2i = [None] * len(ind)
        for i, x in enumerate(ind):
            _n2i[x] = i

        ret = 0
        for i, node in enumerate(ind):
            for cross_node in self._mtx.crosses(node):
                ret += self._mtx.mtx[node, cross_node] * self._route_mtx[i, _n2i[cross_node]]
        return ret//2,

    def __getstate__(self) -> Dict[str, Any]:
        return {}

    def __setstate__(self, _: Dict[str, Any]) -> None:
        self._mtx = g_mtx
        self._route_mtx = g_route_mtx

class IndCreator(object):
    def __init__(self, toolbox: base.Toolbox) -> None:
        self._toolbox = toolbox
    def __call__(self) -> Any:
        return Individual(self._toolbox.random_order())

class EvoAccommodation(object):
    POPULATION_SIZE = 100
    P_CROSSOVER = 0.8
    P_MUTATION = 0.2
    MAX_GENERATIONS = 500
    HALL_OF_FAME_SIZE = 5
    RANDOM_SEED = 44
    WEIGHTS = (-1.,)

    # multiprocess
    PROCESSES = 8

    def __init__(self, mtx: Matrix, manager: SyncManager=None) -> None:
        random.seed(self.RANDOM_SEED)

        self._mtx = mtx
        self._chromo_len = mtx.n

        self._hof = tools.HallOfFame(self.HALL_OF_FAME_SIZE)

        Fitness.patch_weights(self.WEIGHTS)
        # creator.create('FitnessMin', base.Fitness, weights=self.WEIGHTS)
        # creator.create('Individual', list, fitness=creator.FitnessMin)

        self._toolbox = base.Toolbox()
        self._toolbox.register(
            'random_order',
            random.sample, range(self._chromo_len), self._chromo_len,
        )

        self._ind_creator = IndCreator(self._toolbox)
        self._toolbox.register(
            'individual_creator',
            self._ind_creator,
        )
        self._toolbox.register(
            'population_creator',
            tools.initRepeat, list, self._toolbox.individual_creator, self.POPULATION_SIZE,
        )

        self._result = Place2D(self._mtx)

        self._evalator = EvoAccommodationEvalator(self._mtx, self._result.route_mtx)
        self._toolbox.register('evaluate', self._evalator)
        self._toolbox.register('select', tools.selTournament, tournsize=3)
        self._toolbox.register('mate', tools.cxOrdered)
        self._toolbox.register('mutate', tools.mutShuffleIndexes, indpb=1.0/self._chromo_len)

        if manager is not None:
            self._pool = manager.Pool(
                processes=self.PROCESSES,
                initializer=pool_init,
                initargs=(self._mtx, self._result.route_mtx),
            )
            self._toolbox.register(
                'map',
                self._pool.map,
            )

        self._stats = tools.Statistics(lambda ind: np.dot(ind.fitness.values, ind.fitness.weights))
        self._stats.register('min', np.min)
        self._stats.register('avg', np.mean)

    def do(self) -> Place2D:
        population = self._toolbox.population_creator()
        population, logbook = algorithms.eaSimple(population,
            toolbox=self._toolbox,
            cxpb=self.P_CROSSOVER,
            mutpb=self.P_MUTATION,
            ngen=self.MAX_GENERATIONS,
            halloffame=self._hof,
            stats=self._stats,
            verbose=True,
        )

        minStats, _ = logbook.select('min', 'avg')
        print(f'Min avg: [ {", ".join(map(str, minStats))} ]')

        population.sort(key=lambda ind: np.dot(ind.fitness.values, ind.fitness.weights))
        # ret = self._pop2place(population)
        nodes_seq = population[0]

        print(f'Best ind: [ {", ".join(map(str, nodes_seq))} ]')
        print(f'Best fitness: {self._evalator(nodes_seq)[0]}')

        # return ret

    def _pop2place(self, population: List[List[int]]) -> Place2D:
        best_ind = population[0]
        ret = Place2D(self._mtx)
        for i, x in enumerate(best_ind):
            ret.update(node=x, pos=i)
        return ret
