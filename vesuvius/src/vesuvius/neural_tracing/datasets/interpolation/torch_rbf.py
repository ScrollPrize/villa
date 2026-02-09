## from https://github.com/ArmanMaesumi/torchrbf/blob/master/torchrbf/RBFInterpolator.py

import numpy as np
import contextlib
import warnings
import torch
import math

from itertools import combinations_with_replacement
from .radial_fn import SCALE_INVARIANT, RADIAL_FUNCS, MIN_DEGREE

torch.backends.cuda.matmul.allow_tf32 = False

class RBFInterpolator(torch.nn.Module):
    """
    Radial basis function interpolator in Pytorch. This is a port of
    the RBFInterpolator from scipy.interpolate.RBFInterpolator. With
    GPU acceleration, this is much faster than the scipy version.
    SciPy reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html

    @param y: (n, d) tensor of data point coordinates
    @param d: (n, m) tensor of data vectors at y
    @param neighbors (optional): int, specifies the number of neighbors to use 
        for each interpolation point. If None, all points are used.
        Default is None.
    @param neighbors_backend (optional): str, backend for computing nearest neighbors.
        Options are:
        - 'scipy' (uses scipy.spatial.KDTree): Default. Best when GPU memory is a concern.
            Does not support GPU accelerated nearest neighbor search nor autograd. Note that
            scipy backend is usable even when RBFInterpolator is on GPU. Requires scipy installation.
        - 'torch_dense' (uses torch.cdist): Naive O(N^2) nearest neighbor backend.
            Good for small to medium queries on GPU and does not require external
            dependencies. Supports autograd.
        - 'pytorch3d' (uses pytorch3d.ops.knn_points): Best for large queries on GPU.
            Supports autograd. Requires pytorch3d installation.
        Default is 'scipy'.
    @param smoothing (optional): float or (n,) tensor of smoothing parameters
        Default is 0.0.
    @param kernel (optional): str, kernel function to use; one of
        ['linear', 'thin_plate_spline', 'cubic', 'quintic', 'gaussian'
        'multiquadric', 'inverse_multiquadric', 'inverse_quadratic']
        Default is 'thin_plate_spline'.
    @param epsilon (optional): float, shape parameter for the kernel function.
        If kernel is 'linear', 'thin_plate_spline', 'cubic', or
        'quintic', then default is 1.0 and can be ignored. Must be
        specified otherwise.
    @param degree (optional): int, degree of the polynomial added to the
        interpolation function. See scipy.interpolate.RBFInterpolator
        for more details.
    @param device (optional): str, specifies the default device to store tensors
        and perform interpolation.
    @param precision (optional): torch.dtype, specifies the precision used in intermediate
        computations. If None (default), uses the precision of input tensor `d`. 
        If specified, overrides the default precision.

    Returns a callable Torch Module that interpolates the data at given points.
    """

    def __init__(
        self,
        y,
        d,
        neighbors=None,
        neighbors_backend="scipy",
        smoothing=0.0,
        kernel="thin_plate_spline",
        epsilon=None,
        degree=None,
        device="cpu",
        precision=None,
    ):
        super().__init__()

        if torch.backends.cuda.matmul.allow_tf32:
            warnings.warn(
                "TF32 is enabled, which may cause numerical issues in PyTorch RBFInterpolator. "
                "Consider disabling it with torch.backends.cuda.matmul.allow_tf32 = False",
                UserWarning,
            )

        self.device = device

        if precision is None:
            # Use the precision of the d tensor as default
            if isinstance(d, np.ndarray):
                # map numpy dtypes to torch dtypes
                assert d.dtype in [np.float32, np.float64], "numpy dtypes must be float32 or float64"
                self.precision = {
                    np.float32: torch.float32,
                    np.float64: torch.float64,
                }[d.dtype]
            else:
                self.precision = d.dtype
        else:
            assert isinstance(precision, torch.dtype), "precision must be a torch.dtype"
            self.precision = precision

        # init:
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).to(device=device, dtype=self.precision)
        else:
            y = y.to(dtype=self.precision)

        if y.ndim != 2:
            raise ValueError("y must be a 2-dimensional tensor.")

        ny, ndim = y.shape
        if isinstance(d, np.ndarray):
            d = torch.from_numpy(d).to(device=device, dtype=self.precision)
        else:
            d = d.to(dtype=self.precision)

        if d.shape[0] != ny:
            raise ValueError(
                "The first dim of d must have the same length as the first dim of y."
            )

        d_shape = d.shape[1:]
        d = d.reshape((ny, -1))

        if isinstance(smoothing, (int, float)):
            smoothing = torch.full((ny,), smoothing, device=device, dtype=self.precision)
        elif isinstance(smoothing, np.ndarray):
            smoothing = torch.tensor(smoothing, device=device, dtype=self.precision)
        elif not isinstance(smoothing, torch.Tensor):
            raise ValueError("`smoothing` must be a scalar or a 1-dimensional tensor.")
        else:
            smoothing = smoothing.to(dtype=self.precision)

        kernel = kernel.lower()
        if kernel not in RADIAL_FUNCS:
            raise ValueError(f"Unknown kernel: {kernel}")

        if epsilon is None:
            if kernel in SCALE_INVARIANT:
                epsilon = 1.0
            else:
                raise ValueError("Must specify `epsilon` for this kernel.")
        else:
            epsilon = float(epsilon)

        min_degree = MIN_DEGREE.get(kernel, -1)
        if degree is None:
            degree = max(min_degree, 0)
        else:
            degree = int(degree)
            if degree < -1:
                raise ValueError("`degree` must be at least -1.")
            elif degree < min_degree:
                warnings.warn(
                    f"`degree` is too small for this kernel. Setting to {min_degree}.",
                    UserWarning,
                )

        valid_backends = ['scipy', 'torch_dense', 'pytorch3d']
        if neighbors_backend not in valid_backends:
            raise ValueError(f"neighbors_backend must be one of {valid_backends}, got '{neighbors_backend}'")
        
        if neighbors is None:
            nobs = ny
        else:
            neighbors = int(neighbors)
            if neighbors <= 0:
                raise ValueError("neighbors must be a positive integer")
            if neighbors > ny:
                raise ValueError(f"neighbors ({neighbors}) cannot be greater than number of data points ({ny})")
            nobs = neighbors

        powers = monomial_powers(ndim, degree).to(device=device)
        if powers.shape[0] > nobs:
            raise ValueError(f"The data is not compatible with the requested degree. At least {powers.shape[0]} data points are required when degree={degree} and the number of dimensions is {ndim}.")

        if neighbors is None:
            shift, scale, coeffs = solve(y, d, smoothing, kernel, epsilon, powers, self.precision)
            self.register_buffer("_shift", shift)
            self.register_buffer("_scale", scale)
            self.register_buffer("_coeffs", coeffs)
        else:
            # For neighbors mode only precompute shift and scale
            # coeffs need to be computed on the fly in forward pass
            mins = torch.min(y, dim=0).values
            maxs = torch.max(y, dim=0).values
            shift = (maxs + mins) / 2
            scale = (maxs - mins) / 2
            scale[scale == 0.0] = 1.0
            self.register_buffer("_shift", shift)
            self.register_buffer("_scale", scale)
            self._init_neighbor_backend(y, neighbors_backend)

        self.register_buffer("y", y)
        self.register_buffer("d", d)
        self.register_buffer("smoothing", smoothing)
        self.register_buffer("powers", powers)

        self.d_shape = d_shape
        self.neighbors = neighbors
        self.neighbors_backend = neighbors_backend
        self.kernel = kernel
        self.epsilon = epsilon

    def _init_neighbor_backend(self, y, neighbors_backend):
        """Initialize the neighbor engine."""
        assert neighbors_backend in ['scipy', 'torch_dense', 'pytorch3d']
        
        if neighbors_backend == "scipy":
            try:
                from scipy.spatial import KDTree
                y_numpy = y.detach().cpu().numpy()
                self._kdtree = KDTree(y_numpy)
            except ImportError:
                raise ImportError(
                    "scipy is required for neighbors_backend='scipy'. "
                )
        elif neighbors_backend == "pytorch3d":
            try:
                import pytorch3d.ops
                self._knn_points = pytorch3d.ops.knn_points
            except ImportError:
                raise ImportError(
                    "pytorch3d is required for neighbors_backend='pytorch3d'. "
                )
        elif neighbors_backend == "torch_dense":
            return

    def _find_neighbors(self, x):
        """Find k nearest neighbors for each query point in x using specified backend."""
        if self.neighbors_backend == "scipy":
            x_numpy = x.detach().cpu().numpy()
            distances, indices = self._kdtree.query(x_numpy, k=self.neighbors)
            
            if self.neighbors == 1:
                distances = distances.reshape(-1, 1)
                indices = indices.reshape(-1, 1)
            
            indices = torch.from_numpy(indices).to(device=x.device, dtype=torch.long)
            return indices
            
        elif self.neighbors_backend == "torch_dense":
            # Compute distances using pytorch cdist
            distances = torch.cdist(x, self.y)  # (nx, ny)
            _, indices = torch.topk(distances, k=self.neighbors, dim=1, largest=False)
            return indices
            
        elif self.neighbors_backend == "pytorch3d":
            # Use pytorch3d's knn_points to find nearest neighbors
            x_expanded = x.unsqueeze(0)  # (1, nx, ndim)
            y_expanded = self.y.unsqueeze(0)  # (1, ny, ndim)
            # Cast to float32 to avoid errors, also float64 probably not necessary here
            x_expanded = x_expanded.to(dtype=torch.float32)
            y_expanded = y_expanded.to(dtype=torch.float32)
            knn_result = self._knn_points(x_expanded, y_expanded, K=self.neighbors, return_nn=False)
            indices = knn_result.idx.squeeze(0)  # (nx, neighbors)
            return indices

    def forward(self, x: torch.Tensor):
        """
        Returns interpolated data at the given points `x`.

        @param x: (n, d) tensor of points at which to query the interpolator

        Returns a (n, m) tensor of interpolated data.
        """
        if x.dtype != self.precision:
            raise ValueError(f"dtype of `x`, {x.dtype}, does not match RBFInterpolator precision, {self.precision}")

        if x.ndim != 2:
            raise ValueError("`x` must be a 2-dimensional tensor.")

        nx, ndim = x.shape
        if ndim != self.y.shape[1]:
            raise ValueError(
                "Expected the second dim of `x` to have length "
                f"{self.y.shape[1]}."
            )

        if self.neighbors is None:
            kernel_func = RADIAL_FUNCS[self.kernel]

            yeps = self.y * self.epsilon
            xeps = x * self.epsilon
            xhat = (x - self._shift) / self._scale

            kv = kernel_vector(xeps, yeps, kernel_func)
            p = polynomial_matrix(xhat, self.powers)
            vec = torch.cat([kv, p], dim=1)
            out = torch.matmul(vec, self._coeffs)
            out = out.reshape((nx,) + self.d_shape)
            return out
        else:
            return self._forward_neighbors(x)

    def _forward_neighbors(self, x):
        """Forward pass using nearest neighbors for each query point (vectorized)."""
        nx = x.shape[0]
        n_monos = self.powers.shape[0]
        kernel_func = RADIAL_FUNCS[self.kernel]

        neighbor_indices = self._find_neighbors(x)  # (nx, neighbors)
        
        y_neighbors = self.y[neighbor_indices]  # (nx, neighbors, ndim)
        d_neighbors = self.d[neighbor_indices]  # (nx, neighbors, d_flat_size)
        smoothing_neighbors = self.smoothing[neighbor_indices]  # (nx, neighbors)
        
        # Build batched local RBF systems
        lhs_batch, rhs_batch = self._build_batched_local_systems(
            x, y_neighbors, d_neighbors, smoothing_neighbors, kernel_func
        )  # lhs: (nx, system_size, system_size), rhs: (nx, system_size, d_flat_size)
        
        # Solve all systems simultaneously
        try:
            coeffs_batch = torch.linalg.solve(lhs_batch, rhs_batch)  # (nx, system_size, d_flat_size)
        except torch.linalg.LinAlgError:
            # Possible singular matrix, try to nudge diagonal by epsilon:
            eps_diag = 1e-6 * torch.eye(lhs_batch.shape[1], device=lhs_batch.device, dtype=self.precision)
            coeffs_batch = torch.linalg.solve(lhs_batch + eps_diag, rhs_batch)
        
        # Compute interpolated values for all query points
        xeps = x * self.epsilon  # (nx, ndim)
        yeps_neighbors = y_neighbors * self.epsilon  # (nx, neighbors, ndim)
        xhat = (x - self._shift) / self._scale  # (nx, ndim)
        
        kv_batch = self._batched_kernel_vector(xeps, yeps_neighbors, kernel_func)  # (nx, neighbors)
        p_batch = polynomial_matrix(xhat, self.powers)  # (nx, n_monos)
        
        if n_monos > 0:
            vec_batch = torch.cat([kv_batch, p_batch], dim=1)  # (nx, system_size)
        else:
            vec_batch = kv_batch  # (nx, n_neighbors)
        
        out = torch.bmm(vec_batch.unsqueeze(1), coeffs_batch).squeeze(1)  # (nx, d_flat_size)
        
        return out.reshape((nx,) + self.d_shape)

    def _build_batched_local_systems(self, x, y_neighbors, d_neighbors, smoothing_neighbors, kernel_func):
        """Build batched local RBF systems for all query points."""
        nx, n_neighbors, ndim = y_neighbors.shape
        n_monos = self.powers.shape[0]
        system_size = n_neighbors + n_monos
        d_flat_size = d_neighbors.shape[2]
        
        lhs_batch = torch.zeros((nx, system_size, system_size), device=x.device, dtype=self.precision)
        rhs_batch = torch.zeros((nx, system_size, d_flat_size), device=x.device, dtype=self.precision)
        
        yeps_neighbors = y_neighbors * self.epsilon  # (nx, n_neighbors, ndim)
        yeps_expanded1 = yeps_neighbors.unsqueeze(2)  # (nx, n_neighbors, 1, ndim)
        yeps_expanded2 = yeps_neighbors.unsqueeze(1)  # (nx, 1, n_neighbors, ndim)
        pairwise_distances = torch.norm(yeps_expanded1 - yeps_expanded2, dim=3)  # (nx, n_neighbors, n_neighbors)
        
        kernel_matrices = kernel_func(pairwise_distances)  # (nx, n_neighbors, n_neighbors)
        lhs_batch[:, :n_neighbors, :n_neighbors] = kernel_matrices
        diag_indices = torch.arange(n_neighbors, device=x.device)
        lhs_batch[:, diag_indices, diag_indices] += smoothing_neighbors
        
        if n_monos > 0:
            yhat_neighbors = (y_neighbors - self._shift) / self._scale  # (nx, n_neighbors, ndim)
            yhat_flat = yhat_neighbors.reshape(-1, ndim)
            p_matrix_flat = polynomial_matrix(yhat_flat, self.powers)  # (nx * n_neighbors, n_monos)
            p_matrices = p_matrix_flat.reshape(nx, n_neighbors, n_monos)  # (nx, n_neighbors, n_monos)
            lhs_batch[:, :n_neighbors, n_neighbors:] = p_matrices
            lhs_batch[:, n_neighbors:, :n_neighbors] = p_matrices.transpose(1, 2)

        rhs_batch[:, :n_neighbors] = d_neighbors  # (nx, n_neighbors, d_flat_size)
        return lhs_batch, rhs_batch

    def _batched_kernel_vector(self, xeps, yeps_neighbors, kernel_func):
        """Compute kernel vectors for all query points in a batched manner."""
        xeps_expanded = xeps.unsqueeze(1)  # (nx, 1, ndim)
        distances = torch.norm(xeps_expanded - yeps_neighbors, dim=2)  # (nx, n_neighbors)
        kv_batch = kernel_func(distances)  # (nx, n_neighbors)
        
        return kv_batch


def kernel_vector(x, y, kernel_func):
    """Evaluate radial functions with centers `y` for all points in `x`."""
    return kernel_func(torch.cdist(x, y))


def polynomial_matrix(x, powers):
    """Evaluate monomials at `x` with given `powers`"""
    x_ = torch.repeat_interleave(x, repeats=powers.shape[0], dim=0)
    powers_ = powers.repeat(x.shape[0], 1)
    return torch.prod(x_**powers_, dim=1).view(x.shape[0], powers.shape[0])


def kernel_matrix(x, kernel_func):
    """Returns radial function values for all pairs of points in `x`."""
    return kernel_func(torch.cdist(x, x))


def monomial_powers(ndim, degree):
    """Return the powers for each monomial in a polynomial.

    Parameters
    ----------
    ndim : int
        Number of variables in the polynomial.
    degree : int
        Degree of the polynomial.

    Returns
    -------
    (nmonos, ndim) int ndarray
        Array where each row contains the powers for each variable in a
        monomial.

    """
    nmonos = math.comb(degree + ndim, ndim)
    out = torch.zeros((nmonos, ndim), dtype=torch.int32)
    count = 0
    for deg in range(degree + 1):
        for mono in combinations_with_replacement(range(ndim), deg):
            for var in mono:
                out[count, var] += 1
            count += 1

    return out


def build(y, d, smoothing, kernel, epsilon, powers, precision):
    """Build the RBF linear system"""

    p = d.shape[0]
    s = d.shape[1]
    r = powers.shape[0]
    kernel_func = RADIAL_FUNCS[kernel]

    mins = torch.min(y, dim=0).values
    maxs = torch.max(y, dim=0).values
    shift = (maxs + mins) / 2
    scale = (maxs - mins) / 2

    scale[scale == 0.0] = 1.0

    yeps = y * epsilon
    yhat = (y - shift) / scale

    lhs = torch.empty((p + r, p + r), device=d.device, dtype=precision)
    rhs = torch.empty((r + p, s), device=d.device, dtype=precision)
    
    lhs[:p, :p] = kernel_matrix(yeps, kernel_func)
    lhs[:p, p:] = polynomial_matrix(yhat, powers)
    lhs[p:, :p] = lhs[:p, p:].T
    lhs[p:, p:] = 0.0
    lhs[:p, :p] += torch.diag(smoothing)

    rhs[:p] = d
    rhs[p:] = 0.0

    return lhs, rhs, shift, scale


def solve(y, d, smoothing, kernel, epsilon, powers, precision):
    """Build then solve the RBF linear system"""

    lhs, rhs, shift, scale = build(y, d, smoothing, kernel, epsilon, powers, precision)
    try:
        coeffs = torch.linalg.solve(lhs, rhs)
    except RuntimeError:  # singular matrix
        msg = "Singular matrix."
        nmonos = powers.shape[0]
        if nmonos > 0:
            pmat = polynomial_matrix((y - shift) / scale, powers)
            rank = torch.linalg.matrix_rank(pmat)
            if rank < nmonos:
                msg = (
                    "Singular matrix. The matrix of monomials evaluated at "
                    "the data point coordinates does not have full column "
                    f"rank ({rank}/{nmonos})."
                )

        raise ValueError(msg)

    return shift, scale, coeffs