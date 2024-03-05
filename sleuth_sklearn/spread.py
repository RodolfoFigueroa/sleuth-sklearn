import numpy as np
import sleuth_sklearn.stats as st

from numba import njit, prange, types
from sleuth_sklearn.indices import J


@njit(types.f8[:](types.i4[:], types.f8, types.f8), cache=True)
def slope_weight_1d(slope, slp_res, critical_slp):
    """Calculate slope resistance weights.

    Calculates the slope resistance for slope values. The slope
    resistance is a monotonous increasing function of slope boundend in 0-1
    which specific shape is controlled by slp_res coefficient and the
    slope critical value.

    It is interpreted as the probability of resisting urbanization due to
    a steep slope.

    The function is:

    min(1, 1 - (crit_slp - slp)/crit_slp ** 2(slp_res/MAX_SLP_RES))

    Note: The max exponent is 2, which bounds the growth rate of
    rejection probability. A more generall sigmoid function may be
    useful here. Another advantage of a more general sigmoid is that
    the effect of the critical cutoff can be handled by a shape parameter,
    which is differentiable.

    Parameters
    ----------
    slope : np.array
        Slope values for which to calculate resistance.
    slp_res : float
        Coefficient governing the shape of the resistance function.
    critical_slp : float
        Critical slope value above which all weights are one.

    Returns
    -------
    slope_w: np.array
        1D array of slope weights.

    """

    val = (critical_slp - slope) / critical_slp
    exp = 2 * slp_res / 100
    slope_w = np.where(slope >= critical_slp, 0, val)
    slope_w = 1.0 - slope_w**exp
    return slope_w


@njit(types.f8[:, :](types.i4[:, :], types.f8, types.f8), cache=True)
def slope_weight_2d(slope, slp_res, critical_slp):
    """Calculate slope resistance weights.

    Calculates the slope resistance for slope values. The slope
    resistance is a monotonous increasing function of slope boundend in 0-1
    which specific shape is controlled by slp_res coefficient and the
    slope critical value.

    It is interpreted as the probability of resisting urbanization due to
    a steep slope.

    The function is:

    min(1, 1 - (crit_slp - slp)/crit_slp ** 2(slp_res/MAX_SLP_RES))

    Note: The max exponent is 2, which bounds the growth rate of
    rejection probability. A more generall sigmoid function may be
    useful here. Another advantage of a more general sigmoid is that
    the effect of the critical cutoff can be handled by a shape parameter,
    which is differentiable.

    Parameters
    ----------
    slope : np.array
        Slope values for which to calculate resistance.
    slp_res : float
        Coefficient governing the shape of the resistance function.
    critical_slp : float
        Critical slope value above which all weights are one.

    Returns
    -------
    slope_w: np.array
        1D array of slope weights.

    """

    val = (critical_slp - slope) / critical_slp
    exp = 2 * slp_res / 100
    slope_w = np.where(slope >= critical_slp, 0, val)
    slope_w = 1.0 - slope_w**exp
    return slope_w


@njit(types.i4[:, :](types.b1[:, :]), cache=True)
def count_neighbors(Z):
    Z = Z.astype(np.int32)
    N = np.zeros(Z.shape, dtype=np.int32)
    N[1:-1, 1:-1] += (
        Z[:-2, :-2]
        + Z[:-2, 1:-1]
        + Z[:-2, 2:]
        + Z[1:-1, :-2]
        + Z[1:-1, 2:]
        + Z[2:, :-2]
        + Z[2:, 1:-1]
        + Z[2:, 2:]
    )
    return N


@njit(
    types.Tuple((types.i4, types.i4, types.b1[:, :]))(
        types.b1[:, :],
        types.b1[:, :],
        types.i4,
        types.i4,
        types.NumPyRandomGeneratorType("prng"),
        types.f8[:, :],
    ),
    cache=True,
)
def phase1n3_new(
    grid_Z,
    grid_delta,
    # grid_slope,
    # grid_excluded,
    coef_diffusion,
    coef_breed,
    # coef_slope,
    # crit_slope,
    prng,
    # urb_attempt,
    sweights,
):
    """Spontaneus growth and possible new urban centers.

    This function implements the first two stages of growth, the appearance
    of new urban cells at random anywhere on the grid, and the possible
    urbanization of two of their neighbors to create new urban centers.
    Takes the existing urbanizatiojn in the Z grid and writes new urbanization
    into the delta grid.

    Parameters
    ----------
    grid_Z : np.array
        Urban Z grid, stores urbanization from previous year.
    grid_delta : np.array
        Urban delta grid, stores new urbanization for the simulated year.
    grid_slope : np.array
        Array with slope values.
    grid_excluded : np.array
        Array wich labels excluded regions where urbanization is not allowed.
    coef_diffusion : float
        Diffusion coefficient.
    coef_breed : float
        Breed coefficient.
    coef_slope : float
        Slope coefficient.
    crit_slope : float
        Critical slope value above which urbanization is rejected.
    prng : np.random.Generator
        The random number generator.
    urb_attempt : UrbAttempt
        Data class instance to log urbanization attempt metrics.

    Returns
    -------
    sng: int
        Number of newly urbanized pixels during spontaneous growth.
    sdc: int
        Number of newly urbanized pixels corresponding to new urban centers.

    """

    nrows, ncols = grid_Z.shape

    # Get urbanization probability per pixel
    diag = np.sqrt(nrows**2 + ncols**2)
    p_diff = 0.005 * coef_diffusion * diag / (nrows * ncols)
    # Adjust probability with slope rejection
    p_diff = sweights * p_diff

    # Apply urbanization test to all pixels
    # TODO: ignore borders
    mask = prng.random(size=grid_delta.shape) < p_diff
    sng = mask.sum()

    # Update delta grid
    grid_delta = np.logical_or(grid_delta, mask)

    # For PHASE 2
    # kernel = np.array([[1,  1, 1],
    #                    [1,  0, 1],
    #                    [1,  1, 1]])

    # Find number of neighbors in delta grid
    # n_nbrs = convolve(
    #     grid_delta,
    #     kernel,
    #     mode='constant'
    # )
    n_nbrs = count_neighbors(grid_delta)

    # Apply urbanization test
    # Pixels that past the test attempt urbanization
    p_breed = coef_breed / 100.0
    # Adjust probability with slope rejection
    p_breed = sweights * p_breed
    # The probability of urbanization is then (see notes)
    p_urb = (-1) ** (n_nbrs + 1) * (p_breed / 4.0 - 1) ** n_nbrs + 1

    # Apply random test
    mask = prng.random(size=grid_delta.shape) < p_urb
    sdc = mask.sum()

    # Update delta grid
    grid_delta = np.logical_or(grid_delta, mask)

    return sng, sdc, grid_delta


@njit(
    types.Tuple((types.i4, types.b1[:, :]))(
        types.b1[:, :],
        types.b1[:, :],
        types.i4,
        types.NumPyRandomGeneratorType("prng"),
        types.f8[:, :],
    ),
    cache=True,
)
def phase4_new(
    grid_Z,
    grid_delta,
    # grid_slope,
    # grid_excluded,
    coef_spread,
    # coef_slope,
    # crit_slope,
    prng,
    # urb_attempt,
    sweights,
):
    """Edge growth of existing urban clusters composed of 3 or more pixels.

    This function executes edge growth. Each empty pixel with 3 or more
    urbanized neighbors has a probability of urbanize.
    Takes the existing urbanizatiojn in the Z grid and writes new urbanization
    into the delta grid.

    Parameters
    ----------
    grid_Z : np.array
        Urban Z grid, stores urbanization from previous year.
    grid_delta : np.array
        Urban delta grid, stores new urbanization for the simulated year.
    grid_slope : np.array
        Array with slope values.
    grid_excluded : np.array
        Array wich labels excluded regions where urbanization is not allowed.
    coef_spread : float
        Spread coefficient.
    coef_slope : float
        Slope coefficient.
    crit_slope : float
        Critical slope value above which urbanization is rejected.
    prng : np.random.Generator
        The random number generator.
    urb_attempt : UrbAttempt
        Data class instance to log urbanization attempt metrics.

    Returns
    -------
    og: int
        Number of newly urbanized pixels.

    """

    # SLEUTH searches for neighbors of URBANIZED cells, and
    # if two or more neighbors are urbanized, a random neighbor is
    # selected for potential urbanization
    # This incurss in two random tests, the breed coefficient test,
    # and the neighbor sampling. It is possible that the neighbor
    # chosen is already urbanized, and the compound probability
    # of a succesful urbanization of a non-uban pixel is found from
    # p_breed*(# urb_nnbr_clusters) * p_chosen(1/8)
    # On the other hand if we look direcly at the non-urban pixels
    # this is no longer "edge" growth, as its neighbors may not be
    # connected.
    # Here we do it differently, we look only for urban centers, and
    # apply the spread test to urban centers, instead to all urban pixels.
    # That way the interpretation of the spread coef is more closely
    # related to the fraction of urban CENTERS that attempt urbanization,
    # not to the fraction of total urban pixels that attempt urbanization.

    # Loop over pixels or convolution? Original SLEUTH loops over
    # pixels, so convolution can't be worse. But potential improvement
    # if a set of urban pixels is mantained.
    # kernel = np.array([[1,  1, 1],
    #                    [1,  0, 1],
    #                    [1,  1, 1]])

    # Get array with number of neighbors for each pixel.
    # n_nbrs = convolve(
    #     grid_Z,
    #     kernel,
    #     mode='constant'
    # )
    n_nbrs = count_neighbors(grid_Z)

    # Spread centers are urban pixels with 2 or more neighbors
    cluster_labels = np.logical_and(n_nbrs >= 2, grid_Z)

    # Count cluster neighbors per pixel
    # n_nbrs = convolve(
    #     cluster_labels,
    #     kernel,
    #     mode='constant'
    # )
    n_nbrs = count_neighbors(cluster_labels)

    # Apply urbanization test
    # Pixels that past the test attempt urbanization
    p_sprd = coef_spread / 100.0
    # Adjust probability with slope rejection
    p_sprd = sweights * p_sprd
    # The probability of urbanization is then (see notes)
    p_urb = (-1) ** (n_nbrs + 1) * (p_sprd / 8.0 - 1) ** n_nbrs + 1

    # Apply random test
    mask = prng.random(size=grid_delta.shape) < p_urb

    # Update delta grid
    grid_delta = np.logical_or(grid_delta, mask)

    # Get urbanized pixels in this phase
    # Note: this double counts already urbanized pixels.
    og = mask.sum()

    return og, grid_delta


@njit(
    types.bool_[:](
        types.i4[:, :],
        types.b1[:, :],
        types.b1[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4,
        types.i4,
        types.NumPyRandomGeneratorType("prng"),
    ),
    cache=True,
)
def urbanizable(
    coords,
    grid_Z,
    grid_delta,
    grid_slope,
    grid_excluded,
    coef_slope,
    crit_slope,
    prng,
    # urb_attempt
):
    """Determine wether pixels are subject to urbanization.

    Pixels subject to urbanization are not already urbanized pixels that
    pass the tests of slope and the exclution region.

    Parameters
    ----------
    coords : np.array
        Numpy array of pair of (i, j) coordinates of candidate pixels.
    grid_Z: np.array
        2D binary array with original urbanization for the current step.
    grid_delta: np.array
        2D binary array where new urbanization is temporary created.
    grid_slope: np.array
        2D int array with slope values normalized to 1-100 range.
    grid_excluded: np.array
        2D int array with excluded zones. Values are 100*probability of
        rejecting urbanization. 0 is always available, over 100 means
        urbanization  is always rejected.
    coef_slope : float
        Slope coefficient controlling the probability of rejecting
        urbanization due to a steep slope.
    crit_slope : float
        The slope treshold above wich urbanization is always rejected.
    prng : np.random.Generator
        The random number generator class instance.
    urb_attempt : UrbAttempt
        Data class instance to log urbanization attempt metrics.

    Returns
    -------

    mask: np.array
        1D boolean mask for array of candidate coordinates, True if available
        for urbanization.
    """

    # Extract vectors of grid values for candidate pixels.
    z = np.zeros(len(coords), dtype=np.bool_)
    delta = np.zeros(len(coords), dtype=np.bool_)
    slope = np.zeros(len(coords), dtype=np.int32)
    for a, (b, c) in enumerate(coords):
        z[a] = grid_Z[b, c]
        delta[a] = grid_delta[b, c]
        slope[a] = grid_slope[b, c]
    # excld = grid_excluded[ic, jc]

    # Check if not already urbanized in original and delta grid
    z_mask = np.invert(z)
    delta_mask = np.invert(delta)
    # urb_attempt.z_failure += (~z_mask).sum()
    # urb_attempt.delta_failure += (~delta_mask).sum()

    # Apply slope restrictions
    # sweights give the probability of rejecting urbanization
    sweights = slope_weight_1d(slope, coef_slope, crit_slope)
    slp_mask = prng.random(size=len(sweights)) >= sweights
    # urb_attempt.slope_failure += (~slp_mask).sum()

    # Apply excluded restrictions, excluded values >= 100 are
    # completely unavailable, 0 are always available
    # excld value is the 100*probability of rejecting urbanization
    # excld_mask = (prng.integers(100, size=len(excld)) >= excld)
    # urb_attempt.excluded_failure += (~excld_mask).sum()

    mask = np.logical_and(z_mask, delta_mask)
    mask = np.logical_and(mask, slp_mask)

    return mask


@njit(
    types.Tuple((types.i4[:, :], types.bool_[:]))(
        types.i4,
        types.i4,
        types.b1[:, :],
        types.b1[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4,
        types.i4,
        types.NumPyRandomGeneratorType("prng"),
    ),
    cache=True,
)
def urbanizable_neighbors(
    i,
    j,
    grid_Z,
    grid_delta,
    grid_slope,
    grid_excluded,
    coef_slope,
    crit_slope,
    prng,
    # urb_attempt
):
    """Attempt to urbanize the neiggborhood of (i, j).

    Neighbors are chosen in random order until two successful
    urbanizations or all neighbors have been chosen.

    Parameters
    ----------
    i : int
        Row coordinate of center pixel.
    j : int
        Column coordinate of center pixel.
    grid_Z: np.array
        2D binary array with original urbanization for the current step.
    grid_delta: np.array
        2D binary array where new urbanization is temporary created.
    grid_slope: np.array
        2D int array with slope values normalized to 1-100 range.
    grid_excluded: np.array
        2D int array with excluded zones. Values are 100*probability of
        rejecting urbanization. 0 is always available, over 100 means
        urbanization  is always rejected.
    coef_slope : float
        Slope coefficient controlling the probability of rejecting
        urbanization due to a steep slope.
    crit_slope : float
        The slope treshold above wich urbanization is always rejected.
    prng : TODO
        The random number generator class instance.
    urb_attempt : UrbAttempt
        Data class instance to log urbanization attempt metrics.


    Returns
    -------
    nlist: np.array
        Array of neighbor coordinates in random order.
    mask: np.array
       Boolean array for urbanizable neighbors, True if neighbor is
       urbanizable. Same shape as nlist.
    """

    nlist = np.array([i, j], dtype=np.int32) + np.array(
        ((-1, -1), (0, -1), (+1, -1), (+1, 0), (+1, +1), (0, +1), (-1, +1), (-1, 0)),
        dtype=np.int32,
    )
    # TODO: instead of shuffling in place, generate randon indices.
    prng.shuffle(nlist)
    # Obtain urbanizable neighbors
    mask = urbanizable(
        nlist,
        grid_Z,
        grid_delta,
        grid_slope,
        grid_excluded,
        coef_slope,
        crit_slope,
        prng,
        # urb_attempt
    )

    return nlist, mask


@njit(
    types.i4(
        types.b1[:, :],
        types.b1[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4,
        types.i4,
        types.i4,
        types.i4,
        types.i4,
        types.NumPyRandomGeneratorType("prng"),
    ),
    cache=True,
)
def phase5(
    grid_Z,
    grid_delta,
    grid_slope,
    grid_excluded,
    grid_roads,
    grid_roads_dist,
    grid_roads_i,
    grid_roads_j,
    coef_road,
    coef_diffusion,
    coef_breed,
    coef_slope,
    crit_slope,
    prng,
    # urb_attempt
):
    """Road influenced growth.

    This function executes growth influenced by the presence of the
    road network. It looks for urban pixels near a road and attempts to
    urbanize pixels at the road near new urbanizations.
    For each succesful road urbanization, it then attempts to grow a new urban
    center.
    Takes the existing urbanization in the DELTA grid and writes new
    urbanization into the delta grid. The Z grid is still needed as to not
    overwrite previous urbanization unnecessarily.

    Parameters
    ----------
    grid_Z : np.array
        Urban Z grid, stores urbanization from previous year.
    grid_delta : np.array
        Urban delta grid, stores new urbanization for the simulated year.
    grid_slope : np.array
        Array with slope values.
    grid_excluded : np.array
        Array wich labels excluded regions where urbanization is not allowed.
    grid_roads : np.array
        Array with roads labeled by their importance normalized between ??-??.
    grd_roads_dist : np.array
        Array with each pixels distance to the closest road.
    grid_roads_i : np.array
        Array with the i coordinate of the closest road.
    grid_roads_j : np.array
        Array with the j coordinate of the closest road.
    coef_road : float
        Road coefficient.
    coef_diffusion : float
        Diffusion coefficient.
    coef_breed : float
        Breed coefficient.
    coef_slope : float
        Slope coefficient.
    crit_slope : float
        Critical slope value above which urbanization is rejected.
    prng : np.random.Generator
        The random number generator.
    urb_attempt : UrbAttempt
        Data class instance to log urbanization attempt metrics.

    Returns
    -------
    rt: int
        Number of newly urbanized pixels.

    """

    assert coef_road >= 0
    assert coef_diffusion >= 0
    assert coef_breed >= 0

    # calculate tha maximum distance to search for a road
    # this is the chebyshev distance and is precomputed in grid
    # maxed at ~ 1/32 image perimeter
    nrows, ncols = grid_delta.shape
    max_dist = int(coef_road / 100 * (nrows + ncols) / 16.0)
    # number of neighbors up to distance max_dist
    # useless in our code, but original SLEUTH uses this to
    # spiral search the neighborhood of a pixel
    # nneighbors_at_d = 4 * road_gravity * (1 + road_gravity)

    # In this phase we search urbanization on delta grid,
    # not z grid, meaning only new urbanized cells from previous
    # phase are considered. Why?
    # Woudn't it be more apprpriate to keep the infuence of roads constant
    # and influencing non urbanized pixels?
    # The method considers that only new urbanization is influenced by roads.
    # Most of this new urbanization comes from the spontaneous phase,
    # which means we are implicitly chosing pixels at random.

    # From new growth cells, a fraction is selected according to the
    # breed coefficient.
    # In original SLEUTH this is a sample with replacement, which
    # wasteful, and the difference is likely to be absorbed into the
    # calibrated value of the coefficient, so we may want to consider
    # sample without replacement.
    # Though a more thorough analysis of the probabilistic
    # model of SLEUTH is warranted.
    # The problem of sampling without replacement is that having less
    # candidates than breed_coef will result in an error, in such a case,
    # we may return the whole set of candidates.

    # Coordinates of new growth
    w = (grid_delta > 0).nonzero()
    coords = np.stack((w[0], w[1])).T
    # Select n=breed growth candidates

    if len(coords) > 0:
        rand_idx = prng.integers(0, coords.shape[0], size=int(coef_breed) + 1)
        coords = coords[rand_idx]

    # Search for nearest road, and select only if road is close enough
    dists = np.empty(len(coords), dtype=np.int32)
    for a, (b, c) in enumerate(coords):
        dists[a] = grid_roads_dist[b, c]

    coords = coords[dists < max_dist]

    road_i = np.empty(len(coords), dtype=np.int32)
    road_j = np.empty(len(coords), dtype=np.int32)
    for a, (b, c) in enumerate(coords):
        road_i[a] = grid_roads_i[b, c]
        road_j[a] = grid_roads_j[b, c]

    rcoords = np.column_stack((road_i, road_j))

    # For selected roads perform a random walk and attempt urbanization
    # It is perhaps faster justo to choose a road pixel at random and
    # attempt urbanization close to it?
    nlist = np.array(
        (
            (-1, -1),
            (0, -1),
            (+1, -1),
            (+1, 0),
            (+1, +1),
            (0, +1),
            (-1, +1),
            (-1, 0),
            # (0, 0)  # Allows to stay in the same spot, useful if road has no neighbors
        ),
        dtype=np.int32,
    )

    # Here we apply road search as defined in patch_01, which is
    # actually never implmented in official SLEUTH code, despite
    # the official site claiming otherwise.
    # See:
    # http://www.ncgia.ucsb.edu/projects/gig/Dnload/dn_describe3.0p_01.html
    new_sites = np.zeros_like(rcoords)

    for i, rc in enumerate(rcoords):
        for step in range(int(coef_diffusion)):
            prng.shuffle(nlist)
            nbrs = rc + nlist
            flat_idx = np.zeros(len(nbrs), dtype=np.bool_)
            for a, (b, c) in enumerate(nbrs):
                flat_idx[a] = grid_roads[b, c] > 0
            nbrs = nbrs[flat_idx]

            # TODO: there is a bug when a road is a single pixel and has no neighbors
            # TODO: Fix in road preprocessing
            # The following is a hacky patch
            if len(nbrs) == 0:
                break
            rc = nbrs[0]
        new_sites[i] = rc

    # Apply urbanization test based on road weights
    bottom = prng.integers(0, 100)
    mask = np.zeros(len(new_sites), dtype=np.bool_)
    for a, (b, c) in enumerate(new_sites):
        mask[a] = bottom < grid_roads[b, c]
    new_sites = new_sites[mask]

    # Attempt to urbanize
    mask = urbanizable(
        new_sites,
        grid_Z,
        grid_delta,
        grid_slope,
        grid_excluded,
        coef_slope,
        crit_slope,
        prng,
        # urb_attempt
    )
    new_sites = new_sites[mask]
    # Log successes
    rt = len(new_sites)
    # Update available pixels in delta grid
    for a, b in new_sites:
        grid_delta[a, b] = 1

    # Attempt to create new urban centers, urbanize 2 neighbors
    for i, j in new_sites:
        # get urbanizable neighbors
        ncoords, mask = urbanizable_neighbors(
            i,
            j,
            grid_Z,
            grid_delta,
            grid_slope,
            grid_excluded,
            coef_slope,
            crit_slope,
            prng,
            # urb_attempt
        )
        # choose two urbanizable neighbors
        ncoords = ncoords[mask][:2]
        rt += len(ncoords)

        # Update delta grid with values for phase 5
        for a, b in ncoords:
            grid_delta[a, b] = 1

    return rt


@njit(
    types.UniTuple(types.i4, 5)(
        types.b1[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4,
        types.i4,
        types.i4,
        types.i4,
        types.i4,
        types.i4,
        types.NumPyRandomGeneratorType("prng"),
        types.f8[:, :],
    ),
    cache=True,
)
def spread(
    grid_Z,
    grid_slope,
    grid_excluded,
    grid_roads,
    grid_roads_dist,
    grid_roads_i,
    grid_roads_j,
    coef_diffusion,
    coef_breed,
    coef_spread,
    coef_slope,
    coef_road,
    crit_slope,
    prng,
    # urb_attempt,
    sweights,
):
    """Simulate a year's urban growth.

    This function executes urban growth phases for a single step (year).
    Takes urbanization in previous years from the Z grid and temporary
    stores new urbanization in delta grid. Finally updates Z grid at the
    end the the growth phase.

    Parameters
    ----------
    grid_Z : np.array
        Urban Z grid, stores urbanization from previous year.
    grid_slope : np.array
        Array with slope values.
    grid_excluded : np.array
        Array wich labels excluded regions where urbanization is not allowed.
    grid_roads : np.array
        Array with roads labeled by their importance normalized between ??-??.
    grd_roads_dist : np.array
        Array with each pixels distance to the closest road.
    grid_roads_i : np.array
        Array with the i coordinate of the closest road.
    grid_roads_j : np.array
        Array with the j coordinate of the closest road.
    coef_diffusion : float
        Diffusion coefficient.
    coef_breed : float
        Breed coefficient.
    coef_spread : float
        Spread coefficient.
    coef_slope : float
        Slope coefficient.
    coef_road : float
        Road coefficient.
    crit_slope : float
        Critical slope value above which urbanization is rejected.
    prng : np.random.Generator
        The random number generator.
    urb_attempt : UrbAttempt
        Data class instance to log urbanization attempt metrics.

    Returns
    -------
    sng: int
        Number of newly urbanized pixels during spontaneous growth.
    sdc: int
        Number of newly urbanized pixels corresponding to new urban centers.
    og: int
        Number of newly urbanized pixels during edge growth
    rt: int
        Number of newly urbanized pixels during road influenced growth.
    num_growth_pix: int
        Total number of new urban pixels.

    """

    # Initialize delta grid to store temporal urbanization.
    # TODO: zero grid instead of creating
    grid_delta = np.zeros(grid_Z.shape, dtype=np.bool_)

    # Slope coef and crit are constant during a single step
    # TODO:Precalculate all slope weights?

    sng, sdc, grid_delta = phase1n3_new(
        grid_Z=grid_Z,
        grid_delta=grid_delta,
        # grid_slope,
        # grid_excluded,
        coef_diffusion=coef_diffusion,
        coef_breed=coef_breed,
        # coef_slope,
        # crit_slope,
        prng=prng,
        # urb_attempt,
        sweights=sweights,
    )

    og, grid_delta = phase4_new(
        grid_Z=grid_Z,
        grid_delta=grid_delta,
        # grid_slope,
        # grid_excluded,
        coef_spread=coef_spread,
        # coef_slope,
        # crit_slope,
        prng=prng,
        # urb_attempt,
        sweights=sweights,
    )

    rt = phase5(
        grid_Z=grid_Z,
        grid_delta=grid_delta,
        grid_slope=grid_slope,
        grid_excluded=grid_excluded,
        grid_roads=grid_roads,
        grid_roads_dist=grid_roads_dist,
        grid_roads_i=grid_roads_i,
        grid_roads_j=grid_roads_j,
        coef_road=coef_road,
        coef_diffusion=coef_diffusion,
        coef_breed=coef_breed,
        coef_slope=coef_slope,
        crit_slope=crit_slope,
        prng=prng,
        # urb_attempt
    )
    # timers.SPR_PHASE5.stop()

    # Performe excluded test, in this new implmentation
    # this test is only performed once per step in the delta grid
    # excld value is the 100*probability of rejecting urbanization
    # TODO: implement as boolean operation without indexing
    excld_mask = (prng.random(size=grid_delta.shape) * 100) < grid_excluded
    coords = excld_mask.nonzero()
    for a, b in zip(coords[0], coords[1]):
        grid_delta[a, b] = 0

    # Urbanize in Z array for accumulated urbanization.
    # TODO: Try to avoid indexing, implement as boolean operation.
    mask = grid_delta > 0
    coords = mask.nonzero()
    for a, b in zip(coords[0], coords[1]):
        grid_Z[a, b] = grid_delta[a, b]
    # avg_slope = grid_slope[mask].mean()
    num_growth_pix = mask.sum()
    # pop = (grid_Z >= C.PHASE0G).sum()

    return sng, sdc, og, rt, num_growth_pix


@njit(
    types.Tuple((types.i4[:, :, :], types.f8[:, :]))(
        types.b1[:, :],
        types.i8,
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4,
        types.i4,
        types.i4,
        types.i4,
        types.i4,
        types.i4,
        types.NumPyRandomGeneratorType("NumPyRandomGeneratorType"),
    ),
    cache=True,
)
def grow(
    seed_grid,
    nyears,
    grid_slope,
    grid_excluded,
    grid_roads,
    grid_roads_dist,
    grid_roads_i,
    grid_roads_j,
    coef_diffusion,
    coef_breed,
    coef_spread,
    coef_slope,
    coef_road,
    crit_slope,
    prng,
):
    grid_MC = np.zeros((nyears, *seed_grid.shape), dtype=np.int32)
    records = np.empty((nyears, J.TOTAL_SIZE), dtype=np.float64)
    # Initialize Z grid to the seed (first urban grid)
    # TODO: Zero grid instead of creating new one.
    grid_Z = seed_grid.copy()

    # Precalculate/reset slope weighs
    # This can change due to self-modification during growth.
    sweights = 1 - slope_weight_2d(grid_slope, coef_slope, crit_slope)

    for i in range(nyears):
        # Apply CA rules for current year
        sng, sdc, og, rt, num_growth_pix = spread(
            grid_Z=grid_Z,
            grid_slope=grid_slope,
            grid_excluded=grid_excluded,
            grid_roads=grid_roads,
            grid_roads_dist=grid_roads_dist,
            grid_roads_i=grid_roads_i,
            grid_roads_j=grid_roads_j,
            coef_diffusion=coef_diffusion,
            coef_breed=coef_breed,
            coef_spread=coef_spread,
            coef_slope=coef_slope,
            coef_road=coef_road,
            crit_slope=crit_slope,
            prng=prng,
            sweights=sweights,
        )

        # Send stats to current year (ints)
        records[i, J.SNG] = sng
        records[i, J.SDC] = sdc
        records[i, J.OG] = og
        records[i, J.RT] = rt
        records[i, J.NUM_GROWTH_PIX] = num_growth_pix

        # Compute stats
        idx_arr = np.array(
            [
                J.EDGES,
                J.CLUSTERS,
                J.POP,
                J.XMEAN,
                J.YMEAN,
                J.SLOPE,
                J.RAD,
                J.MEAN_CLUSTER_SIZE,
                J.PERCENT_URBAN,
            ]
        )
        records[i, idx_arr] = st.compute_stats(grid_Z, grid_slope)

        # Growth
        records[i, J.GROWTH_RATE] = 100.0 * num_growth_pix / records[i, J.POP]

        # Accumulate MC samples
        # TODO: avoid indexing making sure Z grid is at most 1.
        coords = (grid_Z > 0).nonzero()
        for a, b in zip(*coords):
            grid_MC[i, a, b] += 1

    return (grid_MC, records)


@njit(
    types.Tuple((types.f8[:, :, :], types.f8[:, :], types.f8[:, :]))(
        types.b1[:, :],
        types.i8,  # nyears - DO NOT change to anything other than int/uint
        types.i8,
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4,
        types.i4,
        types.i4,
        types.i4,
        types.i4,
        types.i4,
        types.ListType(types.NumPyRandomGeneratorType("NumPyRandomGeneratorType")),
    ),
    parallel=True,
    cache=True,
)
def fill_montecarlo_grid(
    X0,
    nyears,
    n_iters,
    grid_slope,
    grid_excluded,
    grid_roads,
    grid_roads_dist,
    grid_roads_i,
    grid_roads_j,
    coef_diffusion,
    coef_breed,
    coef_spread,
    coef_slope,
    coef_road,
    crit_slope,
    prngs,
):
    grid_MC = np.zeros((nyears, X0.shape[0], X0.shape[1]), dtype=np.float64)
    records_sum = np.zeros((nyears, J.TOTAL_SIZE), dtype=np.float64)
    records_sum_sq = np.zeros((nyears, J.TOTAL_SIZE), dtype=np.float64)

    for iter in prange(n_iters):
        res = grow(
            seed_grid=X0,
            nyears=nyears,
            grid_slope=grid_slope,
            grid_excluded=grid_excluded,
            grid_roads=grid_roads,
            grid_roads_dist=grid_roads_dist,
            grid_roads_i=grid_roads_i,
            grid_roads_j=grid_roads_j,
            coef_diffusion=coef_diffusion,
            coef_breed=coef_breed,
            coef_spread=coef_spread,
            coef_slope=coef_slope,
            coef_road=coef_road,
            crit_slope=crit_slope,
            prng=prngs[iter],
        )
        grid_MC += res[0]
        records_sum += res[1]
        records_sum_sq += res[1]**2

    grid_MC /= n_iters
    
    records_mean = records_sum / n_iters
    records_std = np.sqrt((records_sum_sq - 2 * records_mean * records_sum) / n_iters + records_mean**2)

    return grid_MC, records_mean, records_std


@njit(
    types.UniTuple(types.f8[:, :, :], 2)(
        types.b1[:, :],
        types.i8,  # nyears - DO NOT change to anything other than int/uint
        types.i8,
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4[:, :],
        types.i4,
        types.i4,
        types.i4,
        types.i4,
        types.i4,
        types.i4,
        types.ListType(types.NumPyRandomGeneratorType("NumPyRandomGeneratorType")),
    ),
    parallel=True,
    cache=True,
)
def run_simulation_io(
    X0,
    nyears,
    n_iters,
    grid_slope,
    grid_excluded,
    grid_roads,
    grid_roads_dist,
    grid_roads_i,
    grid_roads_j,
    coef_diffusion,
    coef_breed,
    coef_spread,
    coef_slope,
    coef_road,
    crit_slope,
    prngs,
):
    grid_MC = np.zeros((nyears, X0.shape[0], X0.shape[1]), dtype=np.float64)
    records = np.zeros((n_iters, nyears, J.TOTAL_SIZE), dtype=np.float64)

    for iter in prange(n_iters):
        res = grow(
            seed_grid=X0,
            nyears=nyears,
            grid_slope=grid_slope,
            grid_excluded=grid_excluded,
            grid_roads=grid_roads,
            grid_roads_dist=grid_roads_dist,
            grid_roads_i=grid_roads_i,
            grid_roads_j=grid_roads_j,
            coef_diffusion=coef_diffusion,
            coef_breed=coef_breed,
            coef_spread=coef_spread,
            coef_slope=coef_slope,
            coef_road=coef_road,
            crit_slope=crit_slope,
            prng=prngs[iter],
        )
        grid_MC += res[0]
        records[iter] = res[1]

    return grid_MC, records


def fill_montecarlo_grid_io(
    X0,
    nyears,
    n_iters,
    grid_slope,
    grid_excluded,
    grid_roads,
    grid_roads_dist,
    grid_roads_i,
    grid_roads_j,
    coef_diffusion,
    coef_breed,
    coef_spread,
    coef_slope,
    coef_road,
    crit_slope,
    prngs,
    log_dir
):
    grid_MC, records = run_simulation_io(
        X0=X0,
        nyears=nyears,
        n_iters=n_iters,
        grid_slope=grid_slope,
        grid_excluded=grid_excluded,
        grid_roads=grid_roads,
        grid_roads_dist=grid_roads_dist,
        grid_roads_i=grid_roads_i,
        grid_roads_j=grid_roads_j,
        coef_breed=coef_breed,
        coef_diffusion=coef_diffusion,
        coef_road=coef_road,
        coef_slope=coef_slope,
        coef_spread=coef_spread,
        crit_slope=crit_slope,
        prngs=prngs,
    )

    grid_MC /= n_iters
    
    out_dir = log_dir / f"{coef_diffusion}_{coef_breed}_{coef_spread}_{coef_slope}_{coef_road}"

    np.save(out_dir / f"records.npy", records)

    records_mean = records.mean(axis=0)
    records_std = records.std(axis=0)

    return grid_MC, records_mean, records_std
