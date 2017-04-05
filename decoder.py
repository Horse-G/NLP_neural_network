import numpy as np
import sys
from collections import defaultdict, namedtuple
from operator import itemgetter

def parse_proj(scores, gold=None):
    '''
    Parse using Eisner's algorithm.

    scores - an (n+1) x (n+1) matrix
    gold - the gold arcs
    '''
    m = len(scores[0])
    '''
    b for backtrack chart
    pi for score chart

    _f for incomplete arc
    _t for complete arc
    '''
    b_f = -np.ones([m, m, 2], dtype=int) 
    b_t = -np.ones([m, m, 2], dtype=int) 
    pi_f = np.zeros([m, m, 2])
    pi_t = np.zeros([m, m, 2]) 

    #initialize incomplete with np.inf
    pi_f[0, :, 0] -= np.inf

    for l in xrange(1,m):
        for i in xrange(m-l):
            j = i+l
            
            # incomplete left arcs
            left_f = pi_t[i, i:j, 1] + pi_t[(i+1):(j+1), j, 0] + scores[j, i] + (0.0 if gold is not None and gold[i]==j else 1.0)
            pi_f[i, j, 0] = np.max(left_f)
            b_f[i, j, 0] = i + np.argmax(left_f)
            # incomplete right arcs
            right_f = pi_t[i, i:j, 1] + pi_t[(i+1):(j+1), j, 0] + scores[i, j] + (0.0 if gold is not None and gold[j]==i else 1.0)
            pi_f[i, j, 1] = np.max(right_f)
            b_f[i, j, 1] = i + np.argmax(right_f)

            #complete left arcs
            left_t = pi_t[i, i:j, 0] + pi_f[i:j, j, 0]
            pi_t[i, j, 0] = np.max(left_t)
            b_t[i, j, 0] = i + np.argmax(left_t)
            #complete right arcs
            right_t = pi_f[i, (i+1):(j+1), 1] + pi_t[(i+1):(j+1), j, 1]
            pi_t[i, j, 1] = np.max(right_t)
            b_t[i, j, 1] = i + 1 + np.argmax(right_t)
        
    value = pi_t[0][m-1][1]
    heads = [-1 for _ in range(m)] 
    backtrack(b_f, b_t, 0, m-1, 1, 1, heads)

    projective_value = 0.0
    for m in xrange(1,m):
        h = heads[m]
        projective_value += scores[h,m]

    return heads


def backtrack(b_f, b_t, i, j, direction, complete, heads):

    '''
    b_f - backtrack chart for incomplete arc
    b_t - backtrack chart for complete arc
    i   - leftmost pointer
    j   - rightmost pointer
    '''
    if i == j:
        return
    if complete:
        d = b_t[i][j][direction]
        if direction == 0:
            backtrack(b_f, b_t, i, d, 0, 1, heads)
            backtrack(b_f, b_t, d, j, 0, 0, heads)
            return
        else:
            backtrack(b_f, b_t, i, d, 1, 0, heads)
            backtrack(b_f, b_t, d, j, 1, 1, heads)
            return
    else:
        d = b_f[i][j][direction]
        if direction == 0:
            heads[i] = j
            backtrack(b_f, b_t, i, d, 1, 1, heads)
            backtrack(b_f, b_t, d+1, j, 0, 1, heads)
            return
        else:
            heads[j] = i
            backtrack(b_f, b_t, i, d, 1, 1, heads)
            backtrack(b_f, b_t, d+1, j, 0, 1, heads)
            return
