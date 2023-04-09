from typing import List
from dataholder import DataHolder
from bucket import BucketIndexer, Bucket
from matrix import Matrix
from iter import do_iter

from alive_progress import alive_bar

from multiprocess.context import SpawnContext
from multiprocess.shared_memory import SharedMemory
from multiprocess.pool import Pool
from multiprocess.queues import Queue

import numpy as np
import argparse
import math

def parse_args():
    parser = argparse.ArgumentParser(prog='lab1')
    parser.add_argument('filename', type=str)
    parser.add_argument('--bucket-cnt', default=0, type=int)
    parser.add_argument('--job-n', default=1, type=int)
    parser.add_argument('--is-test', default=False, action='store_true')
    return parser.parse_args()

def _optimaze_by_size_rec(
    n: int, 
    items: List[int], 
    _ret: List[List[int]]=None,
    _p: List[int]=None, 
    _deep: int=None,
    # optimization
    _dot: int=None,
) -> List[List[int]]:
    # N = a1*p1+...+an*pn = (a, p)
    # p1+...+pn <= ceil(N/a1)
    # p1+...+pn >= floor(N/a2)

    if n < 0:
        # optimization
        return
    elif n == 0:
        _ret.append(_p.copy())
        return

    if _deep is None or _p is None or _ret is None:
        _p = [0] * len(items)
        _ret = []
        _deep = 0
        _dot = 0

    for px in range(int(math.ceil(n/items[_deep])), -1, -1):
        _p[_deep] = px
        dot = _dot + px * items[_deep]

        if _deep + 1 < len(_p):
            _optimaze_by_size_rec(n, items, _ret=_ret, _p=_p, _deep=_deep + 1, _dot=dot)
        elif dot == n:
            _ret.append(_p.copy())
        elif dot < n:
            # optimization
            break
    return _ret 

def optimaze_by_size(n: int, items: List[int]) -> List[List[int]]:
    return _optimaze_by_size_rec(n, items)

def main():
    args = parse_args()

    if args.is_test:
        test()
        return

    print('Start read file...')
    data = DataHolder.by_stream(args.filename)
    print('Reading file has been done')
    
    print('Start calculate bucket sizes combinations...')
    combs = optimaze_by_size(data.get_mtx().n, data.get_bucket_sizes())
    print('Calculation bucket sizes has been done')
    
    total = len(combs)

    mtx = data.get_mtx()

    pool = Pool(processes=args.job_n)
    results = []
    
    print(f'Start iter combs (tolal: {total})...')
    
    with alive_bar(total) as p_bar:
        def _impl(comb):
            indexer = data.forward(comb)
            do_iter(mtx, indexer)
            return indexer
        
        for a in pool.imap(_impl, combs):
            p_bar()
            results.append(a)

    print(f'Sort results...')
    results.sort(key=lambda indexer: indexer.score)
    
    print(f'Best result is "{results[0].score}"')


def test():
    data = DataHolder(Matrix([
            [0, 1, 0, 0, 2, 3, 0, 0],
            [1, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 2, 0, 0, 1, 0],
            [0, 0, 2, 0, 0, 0, 3, 1],
            [2, 0, 0, 0, 0, 1, 0, 0],
            [3, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 3, 0, 0, 0, 2],
            [0, 0, 0, 1, 0, 1, 2, 0],
        ]),
        [4],
    )
    
    mtx = data.get_mtx()
    indexer = BucketIndexer(mtx, [Bucket(4, mtx, [0, 1, 2, 3]), Bucket(4, mtx, [4, 5, 6, 7])]) 

    do_iter(mtx, indexer)

    print(indexer.score)
    

if __name__ == '__main__':
    main()
    # test()