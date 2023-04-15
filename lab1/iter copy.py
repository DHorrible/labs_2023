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

        b_nodes = CapList(cap=bucket.len, data=bucket.nodes)
        for b_node in bucket.nodes:
            if bucket.external_links(b_node) > 0:
                b_nodes.append(b_node)

        while True:
            # Try optimaze
            r = create_bucket_mtx(mtx, bucket, bucket_indexer, externals, b_nodes)

            max_idx = np.unravel_index(r.argmax(), r.shape)
            max_val = int(r[max_idx[0], max_idx[1]])

            old_b_node = b_nodes[max_idx[0]]
            new_b_node = externals[max_idx[1]]
            
            if max_val <= 0:
                break

            b_nodes[max_idx[0]], externals[max_idx[1]] = new_b_node, old_b_node
            
            # print(f'BEFORE: buckket_indexer.score = {bucket_indexer.score}')
            bucket_indexer.swap_nodes(old_b_node, new_b_node)
            # print(f'AFTER: buckket_indexer.score = {bucket_indexer.score}')
            

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
    b_nodes: List[int],
) -> np.ndarray:
    ret = np.zeros((len(b_nodes), len(externals)), dtype=np.int32)

    ext_deltas: List = CapList(len(externals))
    ext_buckets = CapList(len(externals))
    for ext_node in externals:
        other_b_idx = bucket_indexer.n2b_idx(ext_node)
        # ext_deltas[i] = bucket_indexer.at(other_b_idx).links_delta(ext_node)
        ext_deltas.append(bucket.links_delta(ext_node))
        ext_buckets.append(bucket_indexer.at(other_b_idx))

    for b_node_idx, b_node in enumerate(b_nodes):
        for ext_node_idx, ext_node in enumerate(externals):
            b_node_delta = ext_buckets[ext_node_idx].links_delta(b_node)
            ret[b_node_idx, ext_node_idx] = -b_node_delta + -ext_deltas[ext_node_idx] \
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
