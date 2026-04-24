import numpy as np
from scipy.special import binom
from collections import Counter
from itertools import combinations
import numba
from numba import njit

import numpy as np
from scipy.special import binom
from collections import Counter
from itertools import combinations
import numba
from numba import njit

def sort_edge(edge):
    return (min(edge[0], edge[1]), max(edge[0], edge[1]))


#@njit
def List_size_edges(Edges,T_sizes,M_max,M_min):
    order_size= Counter(T_sizes)
    L={}
    for i in range (M_min,M_max+1):
        aux=[]
        for j in range (len(T_sizes)):
            if i==T_sizes[j]:
                aux.append(Edges[j][1])
        L[i]=aux

    return L

#@njit
def interorder_overlap_mn(L, m, n, N):
    """Computes interorder overlap for hyperedges of orders m and n."""
    mn_overlapped = 0
    if len(L[m]) == 0 or len(L[n]) == 0:
        return np.nan
    #print('opa')
    # Convert hyperedges to sorted tuples for fast lookup
    Lm_sorted = [tuple(sorted(e)) for e in L[m]]
    Ln_sorted = [tuple(sorted(e)) for e in L[n]]

    Lm_set = set(Lm_sorted)
    Ln_set = set(Ln_sorted)

    for e_i in Lm_sorted:
        times = 0
        for e_j in Ln_sorted:
            if set(e_i).issubset(set(e_j)):
                if times < 1:
                    mn_overlapped += 1
                times += 1

    possible = set()
    for e_j in Ln_sorted:
        for ele in combinations(e_j, m + 1):
            possible.add(tuple(sorted(ele)))
    print('possible sets', len(possible))
    print('overlaps sets', mn_overlapped)
    return mn_overlapped / len(possible) if len(possible) > 0 else np.nan

def inter_order_overlap_alpha_matrix(H, M_max=None):
    """Computes the inter-order overlap matrix for a hypernetwork."""
    T_sizes = H.edges.size.aslist()
    N = H.num_nodes
    edges_dictionary = H.edges.members(dtype=dict)
    Edges = list(edges_dictionary.items())
    #print(Edges)
    for i in range(len(T_sizes)):
        T_sizes[i] -= 1  # Adjust sizes

    if M_max is None:
        M_max = max(T_sizes)
    
    M_min = min(T_sizes)
    print('Max order:', M_max, 'Min order:', M_min)

    L = List_size_edges(Edges, T_sizes, M_max, M_min)
    alphas = np.full((M_max - M_min, M_max - M_min), np.nan)
    #print(len(alphas))
    for m in range(1, M_max):
        #print(f'{m}/{M_max}')
        for n in range(m + 1, M_max+1):  # Optimize by skipping redundant calculations
            #print(f'm={m}, n={n}')
            alphas = interorder_overlap_mn(L, m, n, N)

    return alphas

def inter_order_overlap_Fede(edges_list, triangles_list):

    # Use a set to store needed links for fast lookup
    needed_links = set()
    print(len(triangles_list))
    # Iterate over triangles to add all unique undirected edges (sorted)
    for triple in triangles_list:
        i, j, k = triple
        needed_links.add(sort_edge((i, j)))
        needed_links.add(sort_edge((i, k)))
        needed_links.add(sort_edge((j, k)))
        
        

    
    # Initialize set to track existing needed links and count duplicates
    existing_needed_links = set()
    repes = 0

    # Iterate over the edges list and check for needed links
    for pair in edges_list:
        edge = sort_edge(pair)
        
        if edge in needed_links:
            if edge not in existing_needed_links:
                existing_needed_links.add(edge)
            else:
                repes += 1

    # Calculate inclusiveness (inter_order_overlap)
    print(len(existing_needed_links),len(needed_links))
    inter_order_overlap = len(existing_needed_links) / len(needed_links)
    
    return inter_order_overlap



#@jit(nopython=True)
def global_intra_order_overlap(N, triangles_list):
    # Efficient calculation of list_k2_nodes
    list_k2_nodes = np.zeros(N, dtype=np.float64)  # Changed to float64
    for a, b, c in triangles_list:
        list_k2_nodes[a] += 1
        list_k2_nodes[b] += 1
        list_k2_nodes[c] += 1

    local_intra_order_overlap = np.ones(N, dtype=np.float64)
    
    for i in range(N):
        k2 = int(list_k2_nodes[i])  # Convert to int for calculations
        if k2 <= 1:
            continue
        
        max_cardinality = 2 * k2 + 1
        min_cardinality = np.ceil((3 + np.sqrt(1 + 8 * k2)) / 2)
        
        union_set = set()
        for triple in triangles_list:
            if i in triple:
                union_set.update(triple)
        
        union_cardinality = len(union_set)
        local_intra_order_overlap[i] = (union_cardinality - min_cardinality) / (max_cardinality - min_cardinality)
    
    return np.dot(1-local_intra_order_overlap, list_k2_nodes) / np.sum(list_k2_nodes)



    