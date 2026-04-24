import numpy as np
import numba
from numba import njit, jit, prange
from numba.typed import Dict
from numba.core import types
import scipy.sparse as sp
import random
from collections import defaultdict


# =============================================================================
# STOCHASTIC SIMULATION: Gillespie algorithm for SIS on hypergraphs
# =============================================================================

@jit(nopython=True)
def SIS_gillespie_temporal_evolution(N, beta1, beta2, mu, iters, edges, triangles,
                                     fixed_init, t_max, check_interval=100,
                                     variance_threshold=1e-4):
    """
    Run multiple Gillespie realisations of the SIS process on a hypergraph
    with pairwise and three-body interactions.

    At each step, the algorithm:
      1. Computes per-node transition rates (infection via edges/triangles + recovery).
      2. Draws an exponential waiting time from the total rate.
      3. Selects the event (node + transition) proportional to individual rates.
      4. Optionally stops early if the prevalence variance drops below a threshold.

    Parameters
    ----------
    N : int
        Number of nodes.
    beta1 : float
        Pairwise infection rate.
    beta2 : float
        Three-body infection rate.
    mu : float
        Recovery rate.
    iters : int
        Number of independent realisations.
    edges : np.ndarray of shape (M, 2)
        Pairwise edges.
    triangles : np.ndarray of shape (T, 3)
        Three-body hyperedges.
    fixed_init : np.ndarray of shape (iters, num_inf)
        Pre-generated initial infected node indices for each realisation.
    t_max : float
        Maximum simulation time.
    check_interval : int
        Number of recent steps used to check for stationarity.
    variance_threshold : float
        Variance threshold below which the simulation is considered stationary.

    Returns
    -------
    times_runs : list of list[float]
        Time points for each realisation.
    I_runs : list of list[int]
        Number of infected nodes at each time point.
    S_runs : list of list[int]
        Number of susceptible nodes at each time point.
    """
    I_runs = []
    S_runs = []
    times_runs = []

    for h in range(iters):
        # Initialise the infection state vector
        indddd = fixed_init[h]
        Infected = np.zeros(N)
        Infected[indddd] = 1

        t = 0
        times = [0]
        I = [np.sum(Infected)]
        S = [N - np.sum(Infected)]

        # Rate vector: stores the total transition rate for each node
        rate_vec = np.zeros(N)

        while t < t_max:
            # --- Compute transition rates ---
            rate_vec.fill(0)

            # Pairwise infection: susceptible node gains rate beta1 per infected neighbour
            for edge in edges:
                v1, v2 = edge
                if Infected[v1] == 0 and Infected[v2] == 1:
                    rate_vec[v1] += beta1
                elif Infected[v2] == 0 and Infected[v1] == 1:
                    rate_vec[v2] += beta1

            # Three-body infection: susceptible node gains rate beta2 when both
            # co-members in a triangle are infected
            for triangle in triangles:
                n1, n2, n3 = triangle
                if Infected[n1] == 0 and Infected[n2] == 1 and Infected[n3] == 1:
                    rate_vec[n1] += beta2
                elif Infected[n2] == 0 and Infected[n1] == 1 and Infected[n3] == 1:
                    rate_vec[n2] += beta2
                elif Infected[n3] == 0 and Infected[n1] == 1 and Infected[n2] == 1:
                    rate_vec[n3] += beta2

            # Recovery: each infected node recovers at rate mu
            rate_vec[Infected == 1] += mu

            # Total rate determines the waiting time
            total_rate = np.sum(rate_vec)
            if total_rate == 0:
                break

            # --- Draw waiting time and advance clock ---
            delay = np.random.exponential(1.0 / total_rate)
            t += delay

            # --- Select which node transitions (proportional to rates) ---
            rr = np.random.random()
            cumulative_rate = 0
            for i in range(N):
                cumulative_rate += rate_vec[i] / total_rate
                if rr < cumulative_rate:
                    # Flip the node's state (S -> I or I -> S)
                    if Infected[i] == 1:
                        Infected[i] = 0
                    elif Infected[i] == 0:
                        Infected[i] = 1
                    break

            # Record the state
            I.append(np.sum(Infected))
            S.append(N - np.sum(Infected))
            times.append(t)

            # --- Early stopping: check if prevalence has reached steady state ---
            if len(I) > check_interval:
                recent_values = np.array(I[-check_interval:]) / N
                variance = np.var(recent_values)
                if variance < variance_threshold:
                    break

        I_runs.append(I)
        times_runs.append(times)
        S_runs.append(S)

    return times_runs, I_runs, S_runs


# =============================================================================
# DEGREE UTILITIES
# =============================================================================

def degree_per_node_list(hyperedges, nodes):
    """
    Compute per-order degree for each node from a list of hyperedges.

    Uses pre-allocated arrays for speed, then converts to dictionaries
    for compatibility with downstream functions.

    Parameters
    ----------
    hyperedges : list of lists/tuples
        Mixed list of hyperedges of varying sizes.
    nodes : list of int
        Node indices to include in the output.

    Returns
    -------
    list of dict
        One dictionary per interaction order, mapping node -> degree.
        Index 0 = order 1 (singletons), index 1 = order 2 (pairs), etc.
    """
    max_order = max(len(edge) for edge in hyperedges)
    node_max = max(nodes)

    # Pre-allocate arrays (much faster than defaultdicts for large networks)
    degree_arrays = [np.zeros(node_max + 1, dtype=np.int32) for _ in range(max_order)]

    for edge in hyperedges:
        order_index = len(edge) - 1
        for node in edge:
            degree_arrays[order_index][node] += 1

    # Convert to dictionaries only for the requested nodes
    return [{node: degree_arrays[i][node] for node in nodes} for i in range(max_order)]


# =============================================================================
# PERFECT HASHING: O(1) lookup for 5D state space (s, i, x, y, z)
#
# The effective degree model tracks nodes by their neighbourhood state
# (s, i, x, y, z) where:
#   s = number of susceptible pairwise neighbours
#   i = number of infected pairwise neighbours
#   x = number of (S,S) triangles
#   y = number of (S,I) or (I,S) triangles
#   z = number of (I,I) triangles
#
# These functions provide collision-free hashing for fast state lookups
# within the Numba-compiled ODE integration.
# =============================================================================

@njit
def perfect_hash_5d(s, i, x, y, z, max_dims):
    """
    Compute a collision-free hash for a 5D state vector.

    Maps (s, i, x, y, z) to a unique integer using the dimension bounds,
    equivalent to flattening a 5D array into a 1D index.
    """
    return (s +
            i * max_dims[0] +
            x * max_dims[0] * max_dims[1] +
            y * max_dims[0] * max_dims[1] * max_dims[2] +
            z * max_dims[0] * max_dims[1] * max_dims[2] * max_dims[3])


@njit
def create_perfect_hash_map(states_array):
    """
    Build a hash map from 5D state vectors to their array indices.

    Parameters
    ----------
    states_array : np.ndarray of shape (n_states, 5)
        Each row is a state (s, i, x, y, z).

    Returns
    -------
    hash_map : numba.typed.Dict
        Maps hash(state) -> index into the values array.
    max_dims : np.ndarray of shape (5,)
        Maximum value + 1 along each dimension (used by the hash function).
    """
    # Determine the bounds of each dimension
    max_dims = np.zeros(5, dtype=np.int64)
    for i in range(len(states_array)):
        for j in range(5):
            if states_array[i, j] > max_dims[j]:
                max_dims[j] = states_array[i, j]
    max_dims += 1  # Include zero

    # Build the hash map: hash(state) -> array index
    hash_map = numba.typed.Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.int64
    )

    for i in range(len(states_array)):
        s, ii, x, y, z = states_array[i]
        hash_key = perfect_hash_5d(s, ii, x, y, z, max_dims)
        hash_map[hash_key] = i

    return hash_map, max_dims


@njit
def find_state_fast(s, i, x, y, z, hash_map, values_array, max_dims):
    """
    Look up the value of a state in O(1) using the perfect hash.

    Returns 0.0 if the state does not exist or has negative indices.
    """
    if s < 0 or i < 0 or x < 0 or y < 0 or z < 0:
        return 0.0

    hash_key = perfect_hash_5d(s, i, x, y, z, max_dims)
    if hash_key in hash_map:
        return values_array[hash_map[hash_key]]
    return 0.0


# =============================================================================
# MOTIF COUNTING: compute neighbourhood statistics for the ODE closure
#
# The effective degree model requires various neighbourhood motif counts
# (e.g. number of S-S edges, I-S-I triangles) and conditional expectations
# (e.g. "given a susceptible node has an infected neighbour, how many of
# its other neighbours are also infected?"). These are computed from the
# current distribution of (s, i, x, y, z) states.
# =============================================================================

@njit
def motifs_eff_degree_optimized(S_states, S_values, I_states, I_values,
                                 S_hash_map, I_hash_map, S_max_dims, I_max_dims):
    """
    Compute motif counts and conditional expectations from the current state
    distribution, used to close the effective degree ODE system.

    Returns
    -------
    motifs1 : tuple of float
        Raw motif counts: (SS, IS, II, SSS, ISS, SIS, ISI, III).
    motifs2 : tuple of float
        Normalised conditional expectations (A, B, C, D, E, F, E2, F2, G, H, J, K)
        used in the derivative computation.
    """
    # --- Initialise raw motif counts ---
    # Pairwise motifs
    SS = IS = II = 0.0
    # Three-body motifs
    SSS = ISS = ISI = SIS = III = 0.0
    ISS2 = 0.0

    # --- Conditional expectations (numerators) ---
    # These track products like i*s, z*s, etc. summed over susceptible nodes,
    # which are then normalised to give the closure approximations
    I_S_S = II_S_S = 0.0          # Pairwise neighbourhood of S nodes
    I_S_SS = II_S_SS = 0.0        # Mixed pairwise-triangle neighbourhood
    I_S_SI = II_S_SI = 0.0        # Triangle neighbourhood correlations
    I_S_I = II_S_I = II_S_II = 0.0  # Higher-order correlations

    # --- Process susceptible nodes ---
    for idx in range(len(S_states)):
        s, i, x, y, z = S_states[idx]
        count = find_state_fast(s, i, x, y, z, S_hash_map, S_values, S_max_dims)

        if count < 1e-12:
            continue

        # Accumulate raw motif counts weighted by the number of nodes in this state
        SS += s * count
        IS += i * count
        SSS += x * count
        SIS += y * count
        ISI += z * count

        # Accumulate products for conditional expectations
        I_S_S += i * s * count
        II_S_S += z * s * count
        I_S_SS += i * x * count
        II_S_SS += z * x * count
        I_S_SI += i * y * count
        II_S_SI += z * y * count

        # Second-order pairwise correlations (require i >= 2)
        if i > 0:
            I_S_I += i * (i - 1) * count

        II_S_I += z * i * count

        # Second-order triangle correlations (require z >= 2)
        if z > 0:
            II_S_II += z * (z - 1) * count

    # --- Process infected nodes (only need II, III, ISS counts) ---
    for idx in range(len(I_states)):
        s, i, x, y, z = I_states[idx]
        count = find_state_fast(s, i, x, y, z, I_hash_map, I_values, I_max_dims)

        if count < 1e-12:
            continue

        II += i * count
        III += z * count
        ISS += x * count

    # --- Normalise to get conditional expectations ---
    # Each ratio approximates "given a node is in motif X, what is the expected
    # state of its remaining neighbours?"
    threshold_denom = 1e-12

    A = I_S_S / SS if SS > threshold_denom else 0.0
    B = II_S_S / SS if SS > threshold_denom else 0.0
    C = I_S_SS / SSS if SSS > threshold_denom else 0.0
    D = II_S_SS / SSS if SSS > threshold_denom else 0.0
    E = I_S_SI / SIS if SIS > threshold_denom else 0.0
    F = II_S_SI / SIS if SIS > threshold_denom else 0.0
    E2 = I_S_SI / ISS if ISS > threshold_denom else 0.0
    F2 = II_S_SI / ISS if ISS > threshold_denom else 0.0
    G = I_S_I / IS if IS > threshold_denom else 0.0
    H = II_S_I / IS if IS > threshold_denom else 0.0
    J = II_S_I / ISI if ISI > threshold_denom else 0.0
    K = II_S_II / ISI if ISI > threshold_denom else 0.0

    motifs1 = SS, IS, II, SSS, ISS, SIS, ISI, III
    motifs2 = A, B, C, D, E, F, E2, F2, G, H, J, K

    return motifs1, motifs2


# =============================================================================
# ODE DERIVATIVES: right-hand side of the effective hyperdegree equations
# =============================================================================

@njit
def compute_derivatives_ultra_fast(S_states, S_values, I_states, I_values,
                                    S_hash_map, I_hash_map, S_max_dims, I_max_dims,
                                    motifs, para):
    """
    Compute the time derivatives dS/dt and dI/dt for all states in the
    effective hyperdegree ODE system.

    The equations track how the number of nodes in each neighbourhood state
    (s, i, x, y, z) evolves due to:
      - Direct infection and recovery of the focal node.
      - Changes in the focal node's neighbourhood caused by infection/recovery
        of its neighbours (mediated by the conditional expectations from motifs).

    Parameters
    ----------
    S_states, S_values : arrays
        States and counts for susceptible nodes.
    I_states, I_values : arrays
        States and counts for infected nodes.
    S_hash_map, I_hash_map : numba.typed.Dict
        Perfect hash maps for O(1) state lookup.
    S_max_dims, I_max_dims : np.ndarray
        Dimension bounds for the hash functions.
    motifs : tuple
        (motifs1, motifs2) from motifs_eff_degree_optimized.
    para : np.ndarray
        [beta1, beta2, gamma] — infection and recovery rates.

    Returns
    -------
    dS_values, dI_values : np.ndarray
        Time derivatives for each state.
    """
    b1, b2, gamma = para
    motifs1, motifs2 = motifs
    A, B, C, D, E, F, E2, F2, G, H, J, K = motifs2

    dS_values = np.zeros_like(S_values)
    dI_values = np.zeros_like(I_values)

    # --- Precompute rate combinations used in the closure equations ---
    # These combine infection rates with the conditional expectations
    b1_A_plus_b2_B = b1 * A + b2 * B                              # S-neighbour infection rate
    b1_C_plus_b2_D = b1 * C + b2 * D                              # SSS-triangle transition rate
    b1_2E_plus_b2_2F = 1.5 * (b1 * E + b2 * F)                   # SIS-triangle transition rate
    b1_plus_b1_G_plus_b2_H = b1 + b1 * G + b2 * H                # I-neighbour infection rate
    b1_E_half_plus_b2_F_half = b1 * E2 + b2 * F2                  # ISS-triangle transition rate
    b2_plus_b1_J_plus_b2_K = b2 + b1 * J + b2 * K                # ISI-triangle transition rate

    # --- Derivatives for susceptible states ---
    for idx in prange(len(S_states)):
        s, i, x, y, z = S_states[idx]
        S_curr = S_values[idx]

        # Corresponding infected count in the same neighbourhood state
        I_curr = find_state_fast(s, i, x, y, z, I_hash_map, I_values, I_max_dims)

        # Look up neighbouring states (shifted by ±1 in each coordinate)
        # These represent the state before/after a single neighbour changes status
        S1 = find_state_fast(s + 1, i - 1, x, y, z, S_hash_map, S_values, S_max_dims) if i > 0 else 0.0
        S2 = find_state_fast(s - 1, i + 1, x, y, z, S_hash_map, S_values, S_max_dims) if s > 0 else 0.0
        S3 = find_state_fast(s, i, x + 1, y - 1, z, S_hash_map, S_values, S_max_dims) if y > 0 else 0.0
        S4 = find_state_fast(s, i, x - 1, y + 1, z, S_hash_map, S_values, S_max_dims) if x > 0 else 0.0
        S5 = find_state_fast(s, i, x, y + 1, z - 1, S_hash_map, S_values, S_max_dims) if z > 0 else 0.0
        S6 = find_state_fast(s, i, x, y - 1, z + 1, S_hash_map, S_values, S_max_dims) if y > 0 else 0.0

        # dS/dt: focal node infection/recovery + neighbour-driven state transitions
        dS_values[idx] = (
            # Focal node: infection (S->I) and recovery (I->S)
            -(b1 * i + b2 * z) * S_curr + gamma * I_curr +
            # Pairwise neighbour becomes infected: s decreases, i increases
            (b1_A_plus_b2_B) * ((s + 1) * S1 - s * S_curr) +
            # Triangle neighbour (SS->SI): x decreases, y increases
            (b1_C_plus_b2_D) * ((x + 1) * S3 - x * S_curr) +
            # Triangle neighbour (SI->II): y decreases, z increases
            (b1_2E_plus_b2_2F) * ((y + 1) * S5 - y * S_curr) +
            # Recovery of neighbours: reverse transitions
            gamma * (-(i + y + 2 * z) * S_curr +
                     (i + 1) * S2 + (y + 1) * S4 + 2 * (z + 1) * S6)
        )

    # --- Derivatives for infected states ---
    for idx in prange(len(I_states)):
        s, i, x, y, z = I_states[idx]
        I_curr = I_values[idx]

        S_curr = find_state_fast(s, i, x, y, z, S_hash_map, S_values, S_max_dims)

        # Neighbouring states (same shift logic as for S)
        I1 = find_state_fast(s + 1, i - 1, x, y, z, I_hash_map, I_values, I_max_dims) if i > 0 else 0.0
        I2 = find_state_fast(s - 1, i + 1, x, y, z, I_hash_map, I_values, I_max_dims) if s > 0 else 0.0
        I3 = find_state_fast(s, i, x + 1, y - 1, z, I_hash_map, I_values, I_max_dims) if y > 0 else 0.0
        I4 = find_state_fast(s, i, x - 1, y + 1, z, I_hash_map, I_values, I_max_dims) if x > 0 else 0.0
        I5 = find_state_fast(s, i, x, y + 1, z - 1, I_hash_map, I_values, I_max_dims) if z > 0 else 0.0
        I6 = find_state_fast(s, i, x, y - 1, z + 1, I_hash_map, I_values, I_max_dims) if y > 0 else 0.0

        # dI/dt: same structure but with different closure coefficients
        # (because the focal node is infected, its neighbours' rates differ)
        dI_values[idx] = (
            # Focal node: infection gain and recovery loss
            (b1 * i + b2 * z) * S_curr - gamma * I_curr +
            # Pairwise neighbour transitions (different rates for I focal node)
            (b1_plus_b1_G_plus_b2_H) * ((s + 1) * I1 - s * I_curr) +
            # Triangle neighbour transitions
            (b1_E_half_plus_b2_F_half) * ((x + 1) * I3 - x * I_curr) +
            (b2_plus_b1_J_plus_b2_K) * ((y + 1) * I5 - y * I_curr) +
            # Recovery of neighbours
            gamma * (-(i + y + 2 * z) * I_curr +
                     (i + 1) * I2 + (y + 1) * I4 + 2 * (z + 1) * I6)
        )

    return dS_values, dI_values


# =============================================================================
# ADAPTIVE TIME STEPPING
# =============================================================================

@njit
def adaptive_timestep(S_values, I_values, dS_values, dI_values, dt_current,
                      tolerance=1e-6, max_dt=0.01, min_dt=1e-8):
    """
    Adjust the integration time step based on the maximum relative change
    in any state variable, ensuring numerical stability.

    Shrinks dt if changes are too large (> 10% relative) and grows it
    if changes are small (< 1% relative).

    Parameters
    ----------
    S_values, I_values : np.ndarray
        Current state values.
    dS_values, dI_values : np.ndarray
        Current derivatives.
    dt_current : float
        Current time step.
    tolerance : float
        Minimum state value to consider for relative change.
    max_dt, min_dt : float
        Bounds on the allowed time step.

    Returns
    -------
    dt_new : float
        Adjusted time step.
    """
    max_rel_change = 0.0

    for i in range(len(S_values)):
        if S_values[i] > tolerance:
            rel_change = abs(dS_values[i] * dt_current / S_values[i])
            if rel_change > max_rel_change:
                max_rel_change = rel_change

    for i in range(len(I_values)):
        if I_values[i] > tolerance:
            rel_change = abs(dI_values[i] * dt_current / I_values[i])
            if rel_change > max_rel_change:
                max_rel_change = rel_change

    # Adjust: shrink if too fast, grow if too slow
    if max_rel_change > 0.1:
        dt_new = max(dt_current * 0.5, min_dt)
    elif max_rel_change < 0.01:
        dt_new = min(dt_current * 1.2, max_dt)
    else:
        dt_new = dt_current

    return dt_new


# =============================================================================
# STATE SPACE INITIALISATION
# =============================================================================

def dict_to_arrays_optimized(state_dict):
    """
    Convert a state dictionary {(s,i,x,y,z): count} into parallel arrays
    for use in Numba-compiled functions.

    Returns
    -------
    states_array : np.ndarray of shape (n_states, 5)
    values_array : np.ndarray of shape (n_states,)
    """
    if not state_dict:
        return np.empty((0, 5), dtype=np.int32), np.empty(0, dtype=np.float64)

    n_states = len(state_dict)
    states_array = np.empty((n_states, 5), dtype=np.int32)
    values_array = np.empty(n_states, dtype=np.float64)

    for i, (state, value) in enumerate(state_dict.items()):
        states_array[i] = state
        values_array[i] = value

    return states_array, values_array


def X_si_xyz_from_degree_list(degree_node, node_list):
    """
    Enumerate all possible neighbourhood states (s, i, x, y, z) reachable
    given the degree sequence of the network.

    For a node with pairwise degree k1 and triangle degree k2, the valid
    states satisfy s + i = k1 and x + y + z = k2.

    Parameters
    ----------
    degree_node : list of dict
        Per-order degree dictionaries (from degree_per_node_list).
    node_list : list of int
        Nodes to consider.

    Returns
    -------
    S_si_xyz, I_si_xyz : dict
        Empty state dictionaries {(s,i,x,y,z): 0} covering all reachable states.
    """
    S_si_xyz = {}
    I_si_xyz = {}

    for node in node_list:
        k1 = degree_node[1][node]  # Pairwise degree
        k2 = degree_node[2][node]  # Triangle degree

        # Enumerate all valid (s, i, x, y, z) combinations
        for i in range(k1 + 1):
            s = k1 - i
            for j in range(k2 + 1):
                for j_ in range(j + 1):
                    x = k2 - j
                    y = j - j_
                    z = j_
                    key = (s, i, x, y, z)
                    if key not in S_si_xyz:
                        S_si_xyz[key] = 0
                        I_si_xyz[key] = 0

    return S_si_xyz, I_si_xyz


def initial_states_fast(initial_infec, node_list, degree_node, edges_node):
    """
    Compute the initial state distribution given a set of initially infected nodes.

    For each node, determines its neighbourhood state (s, i, x, y, z) by
    inspecting the infection status of all its pairwise and triangle neighbours,
    then increments the corresponding S or I state counter.

    Parameters
    ----------
    initial_infec : list of int
        Indices of initially infected nodes.
    node_list : list of int
        All node indices.
    degree_node : list of dict
        Per-order degree dictionaries.
    edges_node : dict
        Maps each node to its list of neighbour groups
        (length 1 = pairwise, length 2 = triangle co-members).

    Returns
    -------
    state_node : dict
        Maps each node to 'S' or 'I'.
    S_si_xyz, I_si_xyz : dict
        Initial state counts {(s,i,x,y,z): count}.
    """
    # Assign initial S/I labels
    state_node = {
        node: 'I' if node in initial_infec else 'S'
        for node in node_list
    }

    # Create empty state dictionaries covering the full state space
    S_si_xyz, I_si_xyz = X_si_xyz_from_degree_list(degree_node, node_list)

    for state in S_si_xyz:
        S_si_xyz[state] = 0
    for state in I_si_xyz:
        I_si_xyz[state] = 0

    # Classify each node's neighbourhood
    for node in node_list:
        s = i = x = y = z = 0

        for neigh in edges_node[node]:
            if len(neigh) == 1:
                # Pairwise neighbour
                v1 = neigh[0]
                if state_node[v1] == 'I':
                    i += 1
                elif state_node[v1] == 'S':
                    s += 1

            elif len(neigh) == 2:
                # Triangle co-members: classify by joint state
                v1, v2 = neigh[0], neigh[1]
                if state_node[v1] == 'S' and state_node[v2] == 'S':
                    x += 1  # Both susceptible
                elif (state_node[v1] == 'S' and state_node[v2] == 'I') or \
                     (state_node[v2] == 'S' and state_node[v1] == 'I'):
                    y += 1  # One infected, one susceptible
                elif state_node[v1] == 'I' and state_node[v2] == 'I':
                    z += 1  # Both infected

        neigh_state = (s, i, x, y, z)

        # Increment the appropriate state counter
        if state_node[node] == 'I':
            I_si_xyz[neigh_state] = I_si_xyz.get(neigh_state, 0) + 1
        elif state_node[node] == 'S':
            S_si_xyz[neigh_state] = S_si_xyz.get(neigh_state, 0) + 1

    return state_node, S_si_xyz, I_si_xyz


# =============================================================================
# MAIN ODE INTEGRATOR
# =============================================================================

def odeint_effective_degree_HO_ultra_optimized(para, hyper_edge_list, N,
                                                initial_infected=None, T=15,
                                                steps_ode=2000, adaptive_dt=True,
                                                progress_callback=None, complete=True):
    """
    Integrate the effective hyperdegree ODE system for the SIS process
    on a hypergraph with pairwise and three-body interactions.

    Uses forward Euler integration with adaptive time stepping, perfect
    hashing for O(1) state lookups, and Numba-compiled kernels for the
    derivative computation.

    Parameters
    ----------
    para : list or array of [beta1, beta2, gamma]
        Infection rates (pairwise, three-body) and recovery rate.
    hyper_edge_list : list of lists
        1-indexed hyperedge list (pairwise edges and triangles combined).
    N : int
        Number of nodes.
    initial_infected : int, list, or None
        Number of initially infected nodes (randomly chosen), a specific
        list of node indices, or None for no initial infection.
    T : float
        Maximum integration time.
    steps_ode : int
        Number of time steps (determines initial dt = T / steps_ode).
    adaptive_dt : bool
        Whether to use adaptive time stepping.
    progress_callback : callable or None
        Optional function called with progress percentage.
    complete : bool
        Unused (kept for API compatibility).

    Returns
    -------
    times : list of float
        Time points.
    S_t : list of float
        Total susceptible population at each time point.
    I_t : list of float
        Total infected population at each time point.
    """
    dt_initial = T / steps_ode
    node_list = list(range(1, N + 1))

    # --- Build per-node neighbour lists from the hyperedge list ---
    # For each node, store the list of "neighbour groups":
    #   - length 1: the other node in a pairwise edge
    #   - length 2: the other two nodes in a triangle
    edges_node = defaultdict(list)
    for edge in hyper_edge_list:
        edge_list = list(edge)
        for node in edge:
            neighbors = [n for n in edge_list if n != node]
            edges_node[int(node)].append(neighbors)

    edges_node = dict(edges_node)
    for node in node_list:
        if node not in edges_node:
            edges_node[node] = []

    # --- Compute per-order degree for each node ---
    degree_node = degree_per_node_list(hyper_edge_list, node_list)

    # --- Set up initial infection ---
    if isinstance(initial_infected, int):
        initial_infected = random.sample(node_list, k=initial_infected)
    elif initial_infected is None:
        initial_infected = []

    # --- Compute initial state distribution ---
    state_node, S_si_xyz, I_si_xyz = initial_states_fast(
        initial_infected, node_list, degree_node, edges_node
    )
    print(f"State space size: {len(S_si_xyz)} susceptible states, {len(I_si_xyz)} infected states")

    # --- Convert to array representation for Numba ---
    S_states, S_values = dict_to_arrays_optimized(S_si_xyz)
    I_states, I_values = dict_to_arrays_optimized(I_si_xyz)

    # Build perfect hash maps for O(1) state lookup
    S_hash_map, S_max_dims = create_perfect_hash_map(S_states)
    I_hash_map, I_max_dims = create_perfect_hash_map(I_states)

    para_array = np.array(para, dtype=np.float64)

    # --- Initialise time series ---
    times = [0.0]
    S_t = [float(N - len(initial_infected))]
    I_t = [float(len(initial_infected))]

    # --- Forward Euler integration with adaptive time stepping ---
    current_time = 0.0
    dt = dt_initial
    step = 0
    max_steps = steps_ode * 10  # Allow extra steps when dt shrinks

    print("Starting integration...")

    while current_time < T and step < max_steps:
        # Compute motif counts and conditional expectations
        motifs = motifs_eff_degree_optimized(
            S_states, S_values, I_states, I_values,
            S_hash_map, I_hash_map, S_max_dims, I_max_dims
        )

        # Compute derivatives for all states
        dS_values, dI_values = compute_derivatives_ultra_fast(
            S_states, S_values, I_states, I_values,
            S_hash_map, I_hash_map, S_max_dims, I_max_dims,
            motifs, para_array
        )

        # Adjust time step for stability
        if adaptive_dt and step > 0:
            dt = adaptive_timestep(S_values, I_values, dS_values, dI_values, dt)

        # Don't overshoot the final time
        if current_time + dt > T:
            dt = T - current_time

        # Forward Euler update
        S_values = S_values + dt * dS_values
        I_values = I_values + dt * dI_values

        current_time += dt
        step += 1

        # Record totals
        S_total = np.sum(S_values)
        I_total = np.sum(I_values)
        times.append(current_time)
        S_t.append(S_total)
        I_t.append(I_total)

        # Progress reporting
        if progress_callback:
            progress_callback(current_time / T * 100)
        elif step % (max_steps // 20) == 0:
            print(f"Progress: {current_time / T * 100:.1f}% (t={current_time:.3f}, dt={dt:.6f})")

        # --- Early termination checks ---
        if I_total < 1e-6:
            print("Infection died out, terminating early")
            break
        if I_total > N * 1.1:
            print("Numerical instability detected, terminating")
            return []

    print(f"Integration completed in {step} steps")
    return times, S_t, I_t


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPER
# =============================================================================

def odeint_effective_degree_HO_optimized(para, hyper_edge_list, N,
                                          initial_infected=None, T=15,
                                          steps_ode=2000, use_numba=True,
                                          use_fast_derivatives=True):
    """
    Wrapper providing backward compatibility with the original API.

    Delegates to odeint_effective_degree_HO_ultra_optimized with
    adaptive time stepping controlled by use_fast_derivatives.
    """
    return odeint_effective_degree_HO_ultra_optimized(
        para, hyper_edge_list, N, initial_infected, T, steps_ode,
        adaptive_dt=use_fast_derivatives
    )