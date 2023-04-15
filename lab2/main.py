import argparse

from multiprocess.managers import SyncManager

import algo as algo

from matrix import Matrix, StreamMatrix
from deap_utils import EvoAccommodation

def parse_args():
    parser = argparse.ArgumentParser(prog='lab2')
    parser.add_argument('filename', type=str)
    parser.add_argument('--job-n', default=1, type=int)
    parser.add_argument('--is-evo', default=False, action='store_true')
    return parser.parse_args()

def test():
    'dummy'

def evo_main(mtx: Matrix, manager: SyncManager) -> None:
    scheme = EvoAccommodation(mtx, manager)
    
    best = scheme.do()

    return best

def main(args, manager: SyncManager=None) -> None:
    if args.filename.lower() == 'test':
        return test()
    
    mtx: Matrix = None
    with open(args.filename, 'r') as f:
        mtx = StreamMatrix(f)

    if args.is_evo:
        return evo_main(mtx, manager)


if __name__ == '__main__':
    args = parse_args()
    if args.job_n > 1:
        with SyncManager() as manager:
            main(args, manager)
    else:
        main(args)
