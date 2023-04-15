from typing import List, Tuple, Set

from llist import dllist

from matrix import Matrix
from bucket import Bucket, BucketIndexer
from cap_list import CapList

import numpy as np

import utils


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
        externals = create_externals(i, bucket_indexer)

        if len(externals) == 0:
            break

        b_nodes: dllist = dllist(list(bucket.nodes))
        while True:
            # utils.dll_remove_if(b_nodes, lambda x: bucket.external_links(x) == 0)
            # utils.dll_remove_if(externals, lambda x: bucket_indexer.n2b(x).external_links(x) == 0)

            # Try optimaze
            r = create_bucket_mtx(mtx, bucket, bucket_indexer, externals, b_nodes)

            max_idx = np.unravel_index(r.argmax(), r.shape)
            max_val = r[max_idx[0], max_idx[1]]

            if max_val <= 0:
                break

            old_b_node = b_nodes.nodeat(int(max_idx[0]))
            new_b_node = externals.nodeat(int(max_idx[1]))

            # print(f'BEFORE: buckket_indexer.score = {bucket_indexer.score}')
            bucket_indexer.swap_nodes(old_b_node.value, new_b_node.value)
            # print(f'AFTER: buckket_indexer.score = {bucket_indexer.score}')
            old_b_node.value, new_b_node.value = new_b_node.value, old_b_node.value

            # update_bucket_mtx(r, mtx, bucket_indexer, externals, b_nodes, new_b_node.value, old_b_node.value)
            

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

    _cache = {}
    def _create_cache_key(b_node, b_idx) -> int:
        return b_node * 10000 + b_idx

    ext_bucket_idxs = CapList(len(externals))
    ext_deltas =  CapList(len(externals))
    for ext_node in externals:
        other_b_idx = bucket_indexer.n2b_idx(ext_node)
        # ext_deltas[i] = bucket_indexer.at(other_b_idx).links_delta(ext_node)
        ext_deltas.append(-bucket.links_delta(ext_node))
        ext_bucket_idxs.append(other_b_idx)

    for b_node_idx, b_node in enumerate(b_nodes):
        for ext_node_idx, ext_node in enumerate(externals):
            b_node_delta = 0
            _cache_key = _create_cache_key(b_node, ext_bucket_idxs[ext_node_idx])
            if _cache_key in _cache:
                b_node_delta = _cache[_cache_key]
            else:
                b_node_delta = -bucket_indexer.at(ext_bucket_idxs[ext_node_idx]).links_delta(b_node)
                _cache[_cache_key] = b_node_delta
            ret[b_node_idx, ext_node_idx] = b_node_delta + ext_deltas[ext_node_idx] \
                - 2 * mtx.mtx[b_node][ext_node]

    return ret

def update_bucket_mtx(
    bucket_matrix: np.ndarray,
    mtx: Matrix,
    bucket_indexer: BucketIndexer,
    externals: List[int],
    b_nodes: List[int],
    old_b_node: int,
    new_e_node: int,
) -> None:
    crossed_b_node_idxs = CapList(len(b_nodes))
    for i, b_node in enumerate(b_nodes):
        if b_node == new_e_node or\
            mtx.mtx[b_node, old_b_node] > 0 or mtx.mtx[b_node, new_e_node] > 0:
            crossed_b_node_idxs.append(i)
            
    bucket = bucket_indexer.n2b(new_e_node)
    ext_bucket = bucket_indexer.n2b(old_b_node)
    crossed_ext_node_idxs = CapList(ext_bucket.len)
    ext_deltas = {}
    for i, ext_node in enumerate(externals):
        if ext_node not in ext_bucket.nodes:
            continue
        if ext_node == old_b_node or\
            mtx.mtx[ext_node, old_b_node] > 0 or mtx.mtx[ext_node, new_e_node] > 0:
            crossed_ext_node_idxs.append(i)
            ext_deltas[ext_node] = -bucket.links_delta(ext_node)

    _cache = {}
    def _create_cache_key(b_node, b_idx) -> int:
        return b_node * 10000 + b_idx

    for b_node_idx in crossed_b_node_idxs:
        b_node = b_nodes[b_node_idx]
        for ext_node_idx in crossed_ext_node_idxs:
            ext_node = externals[ext_node_idx]

            b_node_delta = 0
            ext_bucket_idx = bucket_indexer.n2b_idx(ext_node)
            _cache_key = _create_cache_key(b_node, ext_bucket_idx)
            if _cache_key in _cache:
                b_node_delta = _cache[_cache_key]
            else:
                b_node_delta = -bucket_indexer.at(ext_bucket_idx).links_delta(b_node)
                _cache[_cache_key] = b_node_delta

            bucket_matrix[b_node_idx, ext_node_idx] = b_node_delta + ext_deltas[ext_node] \
                - 2 * mtx.mtx[b_node][ext_node]

def create_externals(
    curr_i: int,
    indexer: BucketIndexer,
) -> List[int]:
    ret: dllist = dllist()
    for i in range(curr_i + 1, indexer.size):
        for x in indexer.at(i).nodes:
            ret.append(x)
    return ret
