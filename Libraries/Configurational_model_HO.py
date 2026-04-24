import numpy as np
import matplotlib.pyplot as plt
import random as random
import xgi as xgi
import networkx as nx
import random
from collections import defaultdict
import numpy as np
from scipy.special import binom
from collections import Counter
from itertools import combinations



def configurational_model_hypergraph(sample_k1, sample_k2, N):
    """
    Generate a random hypergraph using the configuration model with two orders:
    pairwise edges (order 2) and triangles (order 3).

    The algorithm uses stub matching: each node creates stubs equal to its
    target degree, then stubs are randomly paired (for edges) or tripled
    (for triangles) to form hyperedges.

    Parameters
    ----------
    sample_k1 : list of int
        Target pairwise degree for each node.
    sample_k2 : list of int
        Target triangle degree for each node.
    N : int
        Number of nodes.

    Returns
    -------
    edges : list of list[int, int]
        Pairwise edges formed by stub matching.
    triples : list of list[int, int, int]
        Triangles formed by stub matching.
    """

    # =============================================
    # STEP 1: Generate pairwise edges via stub matching
    # =============================================

    # Create a stub list for pairwise interactions
    # Each node i contributes sample_k1[i] copies of itself
    list_stubs = []
    node_list = list(range(N))
    for i in range(len(sample_k1)):
        for k1 in range(sample_k1[i]):
            list_stubs.append(node_list[i])
    random.shuffle(list_stubs)

    edges = []

    # If the total number of stubs is odd, remove one random stub
    # (stubs must pair evenly to form edges)
    if len(list_stubs) % 2 != 0:
        while len(list_stubs) % 2 != 0:
            random_element = random.choice(list_stubs)
            list_stubs.remove(random_element)

    # Repeatedly draw two stubs and form an edge if they belong to different nodes
    # and the edge doesn't already exist (simple graph constraint)
    attempts = 0
    while len(list_stubs) > 0:
        attempts += 1
        if attempts > 10000:
            print('Passed max attempt')
            return

        draw = random.sample(list_stubs, 2)

        # Accept the edge only if it connects two distinct nodes and is not a duplicate
        if draw[1] != draw[0] and draw not in edges:
            edges.append(draw)

        # Remove the drawn stubs regardless of acceptance
        for num in draw:
            list_stubs.remove(num)

    # =============================================
    # STEP 2: Generate triangles via stub matching
    # =============================================

    # Create a stub list for three-body interactions
    # Each node i contributes sample_k2[i] copies of itself
    list_stubs = []
    node_list = list(range(N))
    for i in range(len(sample_k2)):
        for k2 in range(sample_k2[i]):
            list_stubs.append(node_list[i])
    random.shuffle(list_stubs)

    # If the total number of stubs is not divisible by 3, remove random stubs
    # (stubs must group evenly into triples)
    if len(list_stubs) % 3 != 0:
        while len(list_stubs) % 3 != 0:
            random_element = random.choice(list_stubs)
            list_stubs.remove(random_element)

    # Repeatedly draw three stubs and form a triangle if all nodes are distinct
    # and the triangle doesn't already exist
    triples = []
    attempts = 0
    while len(list_stubs) > 0:
        attempts += 1
        if attempts > 10000:
            print('Passed max attempt')
            return

        draw = random.sample(list_stubs, 3)

        # Accept only if all three nodes are distinct and the triple is new
        if draw[0] != draw[1] and draw[0] != draw[2] and draw[1] != draw[2] and draw not in triples:
            triples.append(draw)

        # Remove the drawn stubs regardless of acceptance
        for num in draw:
            list_stubs.remove(num)

    return edges, triples

def higher_order_assortativity_and_correlations(edges, groups):
    """
    Compute assortativity and degree correlations for a higher-order network
    with edges (order-2) and groups of three (order-3).

    Parameters
    ----------
    edges : list of tuple(int, int)
        Pairwise edges.
    groups : list of tuple(int, int, int)
        Groups of three nodes (triads / triangles).

    Returns
    -------
    results : dict
        {
            'pairwise_assortativity': float,
            'triadic_assortativity': float,
            'pairwise_knn': dict(degree -> avg neighbor degree),
            'triadic_knn': dict(hyperdegree -> avg co-member hyperdegree)
        }
    """

    # --- Pairwise degrees ---
    # Build a standard graph from pairwise edges and extract node degrees
    G = nx.Graph()
    G.add_edges_from(edges)
    degree = dict(G.degree())

    # --- Hyperdegrees ---
    # Count how many triangles (order-3 hyperedges) each node participates in
    hyperdegree = defaultdict(int)
    for g in groups:
        for node in g:
            hyperdegree[node] += 1

    # --- Pairwise assortativity ---
    # Measures degree-degree correlation across standard edges
    # Positive = assortative (high-degree nodes connect to high-degree), negative = disassortative
    pairwise_assort = nx.degree_assortativity_coefficient(G)

    # --- Triadic assortativity ---
    # Measures hyperdegree-hyperdegree correlation across triangles
    # For each triangle, consider all pairs of nodes and compare their hyperdegrees
    triad_pairs = []
    for g in groups:
        triad_pairs.extend(combinations(g, 2))

    deg_u = [hyperdegree[u] for u, v in triad_pairs]
    deg_v = [hyperdegree[v] for u, v in triad_pairs]
    # Pearson correlation between hyperdegrees of co-members in triangles
    triadic_assort = np.corrcoef(deg_u, deg_v)[0, 1] if len(triad_pairs) > 0 else np.nan

    # --- Pairwise k_nn(k) ---
    # Average nearest-neighbour degree as a function of node degree (pairwise layer)
    # For each edge, record the neighbour's degree grouped by the focal node's degree
    knn_pairwise = defaultdict(list)
    for u, v in edges:
        knn_pairwise[degree[u]].append(degree[v])
        knn_pairwise[degree[v]].append(degree[u])
    knn_pairwise_avg = {k: np.mean(vals) for k, vals in knn_pairwise.items()}

    # --- Triadic k_nn(k) ---
    # Same idea but for the triangle layer: average co-member hyperdegree
    # as a function of the focal node's hyperdegree
    knn_triadic = defaultdict(list)
    for g in groups:
        for u, v in combinations(g, 2):
            knn_triadic[hyperdegree[u]].append(hyperdegree[v])
            knn_triadic[hyperdegree[v]].append(hyperdegree[u])
    knn_triadic_avg = {k: np.mean(vals) for k, vals in knn_triadic.items()}

    return {
        'pairwise_assortativity': pairwise_assort,
        'triadic_assortativity': triadic_assort,
        'pairwise_knn': knn_pairwise_avg,
        'triadic_knn': knn_triadic_avg
    }


def get_degree_list_from_edges(edges, max_order=2, N=100):
    """
    Compute per-order degree sequences from a list of hyperedges.

    Parameters
    ----------
    edges : list of tuples
        Mixed list of pairwise edges (length 2) and triangles (length 3).
    max_order : int
        Maximum interaction order to consider (2 = pairwise + triangles).
    N : int
        Total number of nodes in the network.

    Returns
    -------
    degree_list : list of dicts
        degree_list[0] = {node: pairwise degree}, degree_list[1] = {node: triangle degree}
    """
    node_list = range(N)
    degree_list = []

    # Initialise a degree dictionary for each order, with all nodes set to 0
    for order in range(0, max_order):
        degree_list_order = {}
        for node in node_list:
            degree_list_order[node] = 0
        degree_list.append(degree_list_order)

    # Iterate over all hyperedges and increment the appropriate order's degree
    # Order is determined by hyperedge size: size 2 -> order 0, size 3 -> order 1
    for edge in edges:
        order = len(edge) - 2
        for ele in edge:
            degree_list[order][ele] += 1

    return degree_list


def generate_powerlaw_degrees(n, gamma, min_degree=1, max_degree=None):
    """
    Generate a power-law distributed degree sequence using inverse transform sampling.

    Parameters
    ----------
    n : int
        Number of nodes.
    gamma : float
        Power-law exponent (P(k) ~ k^{-gamma}).
    min_degree : int
        Minimum allowed degree.
    max_degree : int or None
        Maximum allowed degree (defaults to n).

    Returns
    -------
    degrees : np.array of int
        Sampled degree sequence with even sum (required for stub matching).
    """
    from scipy import stats

    if max_degree is None:
        max_degree = n

    # scipy's Pareto distribution: P(x) ~ x^{-(a+1)}, so a = gamma - 1
    a = gamma - 1
    degrees = stats.pareto.rvs(a, scale=min_degree, size=n)
    degrees = np.clip(degrees, min_degree, max_degree).astype(int)

    # Ensure the total number of stubs is even (needed to form complete edges)
    if sum(degrees) % 2 == 1:
        degrees[0] += 1

    return degrees


'''Define some auxiliary functions for this notebook'''

    
def plot_hyperdegree_summary(hyper_edges, N, kernel=True,title=None):
    from scipy.stats import poisson, gaussian_kde
    from scipy import stats

    """
    Plot degree distributions and degree-degree correlation for a hypergraph.
    
    Parameters:
        hyper_edges: list of edges and hyperedges
        N: number of nodes
        kernel: if True, plot 2D KDE density; if False, plot scatter plot
    """
    # Extract per-order degree lists from the hyperedges
    degree_list = get_degree_list_from_edges(hyper_edges, N=N)
    degrees1 = list(degree_list[0].values())
    degrees2 = list(degree_list[1].values())

    # --- Three-panel figure: two degree distributions + correlation panel ---
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    colors = ['coral', 'skyblue']
    order_labels = ['Pairwise', 'Three-body']
    if title is not None:
        fig.suptitle(title, fontsize=16)
    # Panels 0 and 1: degree distribution histograms
    for i in range(2):
        degrees = list(degree_list[i].values())

        bins = np.arange(min(degrees) - 0.5, max(degrees) + 1.5, 1)

        axs[i].hist(degrees, bins=bins, color=colors[i], alpha=0.5, density=True, edgecolor='black')

        mean_k = np.mean(degrees)
        var_k = np.var(degrees)
        max_k = max(degrees)
        print(f'Average k_{i+1} = {round(mean_k, 2)}')
        print(f'Variance k_{i+1} = {round(var_k, 2)}')
        print(f'Maximum k_{i+1} = {max_k}')

        axs[i].set_title(f'{order_labels[i]} degree distribution', fontsize=14)
        axs[i].set_ylabel('Probability', fontsize=13)
        axs[i].set_xlabel('Degree', fontsize=13)
        axs[i].set_xlim(0.5, 6.5)
        axs[i].set_xticks(range(1, 7))
        axs[i].tick_params(axis='both', labelsize=13)

    # Panel 2: degree-degree correlation
    k1_arr = np.array(degrees1)
    k2_arr = np.array(degrees2)

    if kernel:
        # 2D kernel density estimate
        jitter = 0.05
        k1_jittered = k1_arr + np.random.normal(0, jitter, len(k1_arr))
        k2_jittered = k2_arr + np.random.normal(0, jitter, len(k2_arr))

        kde = gaussian_kde(np.vstack([k1_jittered, k2_jittered]))

        x_grid = np.linspace(k1_arr.min() - 1, k1_arr.max() + 1, 200)
        y_grid = np.linspace(k2_arr.min() - 1, k2_arr.max() + 1, 200)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

        im = axs[2].contourf(X, Y, Z, levels=20, cmap='plasma')
        fig.colorbar(im, ax=axs[2], label='Density')
        axs[2].set_title('Hyperdegree density', fontsize=14)
    else:
        # Simple scatter plot
        axs[2].scatter(k1_arr, k2_arr, color='lightgreen', alpha=0.5, edgecolors='black', linewidths=0.5)
        axs[2].set_title('Hyoerdegrees scatter', fontsize=14)

    axs[2].set_xlabel(r'$k_1$', fontsize=13)
    axs[2].set_ylabel(r'$k_2$', fontsize=13)
    axs[2].tick_params(axis='both', labelsize=13)

    plt.tight_layout()
    plt.show()

    # Pearson correlation between the two degree sequences
    r, p_value = stats.pearsonr(degrees1, degrees2)
    print(f'Pearson correlation: r = {round(r, 4)}, p-value = {round(p_value, 4)}')
    return 

def configurational_model_hypergraph_correlated(dist_x, dist_y, rho, size, return_hypergraph=True):
    from scipy.stats import nbinom, norm
    """
    Generate correlated samples from arbitrary scipy.stats distributions
    using a Gaussian copula.

    Parameters
    ----------
    dist_x : scipy.stats frozen distribution  (e.g. stats.nbinom(r, p))
    dist_y : scipy.stats frozen distribution  (e.g. stats.expon(scale=2))
    rho    : float in [-1, 1], copula correlation
    size   : int, number of samples

    Returns
    -------
    X, Y : np.ndarray, np.ndarray
    """
    # Correlated standard normals
    Z = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size)

    # Normal CDF → uniform marginals
    U = norm.cdf(Z)

    # Inverse CDF of each target distribution
    X = dist_x.ppf(U[:, 0])
    Y = dist_y.ppf(U[:, 1])


    X = X.astype(int)
    Y = Y.astype(int)

    N=len(X)
        
    if return_hypergraph:
        return configurational_model_hypergraph(X, Y, N)
    else:
        return X,Y

def plot_hypergraph_with_degree(H):
    pos = xgi.barycenter_spring_layout(H, seed=1)

    fig, ax = plt.subplots(figsize=(6, 2.5))

    ax, collections = xgi.draw(
        H,
        pos=pos,
        node_fc=H.nodes.degree(order=2),
        node_size=H.nodes.degree(order=1),
        node_fc_cmap="Reds",
    )

    node_col, _, edge_col = collections

    plt.colorbar(node_col, label="Degree (order 2)")

    # --- Size legend for degree order 1 ---
    degree1_values = H.nodes.degree(order=1).aslist()
    
    # XGI internally maps node_size through a linear rescaling
    # to a default range (roughly 15–75 area units). Inspect the
    # actual sizes from the drawn collection:
    rendered_sizes = node_col.get_sizes()

    # Build a mapping from degree -> rendered size
    degree_to_rendered = dict(zip(degree1_values, rendered_sizes))

    # Pick a few representative values
    unique_degrees = sorted(set(degree1_values))
    if len(unique_degrees) > 5:
        # Sample evenly across the range
        indices = np.linspace(0, len(unique_degrees) - 1, 5, dtype=int)
        legend_degrees = [unique_degrees[i] for i in indices]
    else:
        legend_degrees = unique_degrees

    for d in legend_degrees:
        ax.scatter(
            [], [],
            s=degree_to_rendered[d],
            c="grey",
            alpha=0.6,
            label=f"{d}",
        )

    ax.legend(
        title="Degree (order 1)",
        labelspacing=1.5,
        borderpad=1.2,
        frameon=True,
        loc="lower right",
        scatterpoints=1,
    )

    plt.tight_layout()
    plt.show()