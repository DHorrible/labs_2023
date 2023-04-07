import copy

from typing import List, Tuple, Set

from matrix import Matrix
from bucket import Bucket, BucketIndexer
from cap_list import CapList

import numpy as np


matrixTest = [[0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 1, 0],
              [0, 1, 0, 0, 0, 1, 0, 1],
              [0, 0, 1, 0, 0, 0, 1, 0],
              [1, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 1, 0, 0, 1],
              [0, 1, 0, 1, 0, 0, 1, 0]]

bucketsTest = [[0, 1],
               [2, 3, 4],
               [5, 6, 7]]


def do_iter(
    mtx: Matrix,
    bucket_indexer: BucketIndexer,
) -> None:
    for i, bucket in enumerate(bucket_indexer.buckets):
        externals = create_externals(i, bucket_indexer, _hint_n=mtx.n)

        if len(externals) == 0:
            break

        while True:            
            b_nodes = CapList(cap=bucket.len)
            for b_node in bucket.nodes:
                if bucket.external_links(b_node) > 0:
                   b_nodes.append(b_node) 

            # Try optimaze
            r = create_bucket_mtx(mtx, bucket, bucket_indexer, externals)

            max_idx = np.unravel_index(r.argmax(), r.shape)
            max_val = r[max_idx[0], max_idx[1]]

            old_b_node = b_nodes[max_idx[0]]
            new_b_node = externals[max_idx[1]]
            
            if max_val <= 0:
                break

            b_nodes[max_idx[0]], externals[max_idx[1]] = new_b_node, old_b_node
            bucket_indexer.swap_nodes(old_b_node, new_b_node)



# def swap_np_mtx(
#     mtx: np.ndarray,
#     x: int,
#     y: int,
# ) -> None:
#     for i in range(len(mtx)):
#         mtx[x][i], mtx[y][i] = mtx[y][i], mtx[x][i]
#     mtx[x], mtx[y] = mtx[y], mtx[x]
    

def create_bucket_mtx(
    mtx: Matrix,
    bucket: Bucket,
    bucket_indexer: BucketIndexer,
    externals: List[int],
) -> np.ndarray:
    ret = np.ndarray((bucket.len, len(externals)))
    
    ext_deltas = [None] * len(externals)
    for i in range(len(ext_deltas)):
        ext_node = externals[i]
        other_b_idx = bucket_indexer.n2b(ext_node)
        ext_deltas[i] = bucket_indexer.at(other_b_idx).links_delta(ext_node)

    for b_node_idx, b_node in enumerate(bucket.nodes):
        b_node_delta = bucket.links_delta(b_node)
        for ext_node_idx, ext_node in enumerate(externals):
            ret[b_node_idx, ext_node_idx] = b_node_delta + ext_deltas[ext_node_idx]\
                - 2 * mtx.mtx[b_node][ext_node]

    return ret

def create_externals(
    curr_i: int,
    indexer: BucketIndexer,
    _hint_n: int=0,
) -> List[int]:
    ret = CapList(cap=_hint_n)
    for i in range(curr_i + 1, indexer.size):
        bucket = indexer.at(i)
        for x in bucket.nodes:
            if bucket.external_links(x) > 0:
                ret.append(x)
    return ret
