"""
Data types for topic decomposition (:mod:`topic3d.datatypes`)
=====================================================

.. authors:: Gautham Ganesh, Jean-Bernard Fiche, Marcelo Nollmann

Function reference
------------------
"""


from numbers import Number
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.linalg import issymmetric
import warnings
import computation as cmp
import visualization


def normalize_minmax(arr:np.ndarray) -> np.ndarray:
    """ Normalize the values of an array to the range of [0, 1] using Min-Max Normalization
    
    Parameters
    ----------
    arr : an array

    Returns
    ----------
    arr_rescaled : rescaled input array
    """

    arr_min = np.min(arr)
    arr_rescaled = (arr - arr_min) / (np.max(arr) - arr_min)
    
    return arr_rescaled


def regularize_matrix(matrix:np.ndarray, hack=False) -> np.ndarray:
    """ Regularize the given matrix to make it positive-definite, i.e., having no negative eigenvalues
    
    Parameters
    ----------
    matrix : two-dimensional square array\n
    hack : if `True`, the lowest positive eigenvalue will be replaced by a small positive value

    Returns
    ----------
    reg_matrix : regularised matrix
    """

    assert matrix.ndim == 2
    N = matrix.shape[0]

    eigenvals, eigenvecs = np.linalg.eig(matrix)
    if np.all(eigenvals > 0) and hack:
        eigenvals[np.argmin(eigenvals)] = 1e-06
    else:
        eigenvals[eigenvals < 0] = 1e-06
    reg_matrix = eigenvecs @ np.diag(eigenvals) @ np.linalg.inv(eigenvecs)
    
    return reg_matrix


########################

def rescale_minmax(arr:np.ndarray, ref:np.ndarray) -> np.ndarray:

    """
    Rescale the values of an array `arr` to the range of another array `ref` using Min-Max Normalization
    
    Parameters
    ----------
    arr, ref : two arrays of the same shape

    Returns
    ----------
    arr_rescaled : rescaled input array
    """
    
    assert np.logical_not(np.isnan(arr).all())

    ref = ref[np.logical_not(np.isnan(ref))]
    new_min = np.min(ref)
    new_max = np.max(ref)

    arr_rescaled = (normalize_minmax(arr) * (new_max - new_min)) + new_min

    return arr_rescaled



class CoordsArrayStack(np.ndarray):

    """
    A stack of arrays of coordinates and their metadata, used primarily for the storage of traces from chromatin 
    tracing experiments and coarse-grained polymer simulations

    Parameters
    --------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.

    See Also
    --------
    average : Weighted average
    std, var, nanmean, nanstd, nanvar

    Notes
    --------
    The arithmetic mean is the sum of the elements along the axis divided
    by the number of elements.
    """

    # Initializing using __new__ instead of __init__ in line with Numpy
    def __new__(cls, arr:np.ndarray):

        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 2 or 3
        assert arr.shape[-1] == 3

        if arr.ndim == 2:
            arr = arr[np.newaxis, :]

        obj = np.asarray(arr).view(cls)

        # add new attributes to the created instance
        obj.num_items = arr.shape[0]
        obj.num_bc = arr.shape[1]
        return obj


    # Since new attributes were added in __new__
    def __array_finalize__(self, obj):
        # reset the attribute from passed original object
        self.num_items = getattr(obj, "num_items", None)
        self.num_bc = getattr(obj, "num_bc", None)


    def compute_CoM(self) -> np.ndarray:
        """Compute the center of mass (CoM) of each array in the stack"""

        return np.nanmean(self, axis=-2)


    def compute_pwds(self):
        """Compute pairwise distances for each array in the stack"""

        pwds = np.empty((self.num_items, int((self.num_bc * (self.num_bc - 1)) / 2), ))
        for i in range(self.num_items):
            pwds[i] = pdist(self[i])
        pwds = PairwiseArrayStack(pwds)
        
        return pwds


    def compute_bc_effs(self) -> np.ndarray:
        """Compute barcode hybridization efficiency (percentage of barcode detection)"""

        bc_effs = np.zeros((self.num_bc,))
        for i in range(self.num_bc):
            num_bc_found = np.sum(~np.isnan(self[:, i, :]).any(axis=1))
            bc_effs[i] = np.around(num_bc_found * 100 / self.num_items, decimals=1)

        return bc_effs


    # TODO more efficient approach for the naive method
    def impute(self, method="naive"):
        """ Impute missing barcode coordinates (detected by a NaN) with the mean position of the neighboring barcodes. 
        Barcode coordinates at either end of the trace cannot be imputed.

        Parameters
        --------
        method : str
            naive for neighbor mean imputation
            jost for Daniel Jost's method using conditional multivariate normal distributions for the coordinates

        Returns
        -------
        stack_imputed : CoordsArrayStack
            imputed stack
        """

        method = method.lower()
        assert method == "naive"    # in ["naive", "jost"]

        if ~np.all(np.isnan(self)):
            warnings.warn("No NaNs to impute in the stack")
            return self
        
        if method == "naive":
            neighbor_means = (self[:, 2:] + self[:, :-2]) / 2   # compute the mean of all potential neighbors in order
            stack_imputed = self[:]
            is_nan = np.isnan(stack_imputed[:, 1:-1])
            stack_imputed[:, 1:-1][is_nan] = neighbor_means[np.where(is_nan)]   # replace NaNs with mean positions

        elif method == "jost":
            stack_imputed = self[:]
            for n in range(self.num_items):
                tr = stack_imputed[n][:]
                mean2 = pdist(tr) ** 2
                assert mean2.ndim == 2
                assert tr.shape[-1] == 3
                assert mean2.shape[0] == self.num_bc
                # missing positions
                missing_pos_bool = np.isnan(tr[:,0])
                available_pos_bool = np.logical_not(missing_pos_bool)
                s = np.sum(available_pos_bool)
                # reorganizing each axis' coordinates
                x = tr[:,0]
                y = tr[:,1]
                z = tr[:,2]
                x_reorg = np.concatenate((x[available_pos_bool], x[missing_pos_bool]))
                y_reorg = np.concatenate((y[available_pos_bool], y[missing_pos_bool]))
                z_reorg = np.concatenate((z[available_pos_bool], z[missing_pos_bool]))
                # reorganizing matrix as well
                missing_idxs = np.argwhere(missing_pos_bool).T[0]
                available_idxs = np.argwhere(available_pos_bool).T[0]
                new_order = np.concatenate((available_idxs, missing_idxs))
                mean2_reorg = reorganize(mean2, new_order)
                # get sigma matrix
                sigma = cmp.compute_cov_matrix(mean2_reorg)
                mu = np.zeros((1, self.num_bc-1))
                mu_available = mu[:,:s-1]
                mu_missing = mu[:,s-1:]
                sigma_nz_nz = sigma[0:s-1, 0:s-1]
                sigma_z_z = sigma[s-1:, s-1:]
                sigma_z_nz = sigma[s-1:, 0:s-1]
                sigma_nz_z = sigma[0:s-1, s-1:]
                # infer conditional mean and sigma
                mucondx = mu_missing + (sigma_z_nz @ (np.linalg.inv(sigma_nz_nz) @ ((x_reorg[1:s] - x_reorg[0]) - mu_available).T)).T
                mucondy = mu_missing + (sigma_z_nz @ (np.linalg.inv(sigma_nz_nz) @ ((y_reorg[1:s] - y_reorg[0]) - mu_available).T)).T
                mucondz = mu_missing + (sigma_z_nz @ (np.linalg.inv(sigma_nz_nz) @ ((z_reorg[1:s] - z_reorg[0]) - mu_available).T)).T
                sigcond = sigma_z_z - ((sigma_z_nz @ np.linalg.inv(sigma_nz_nz)) @ sigma_nz_z)
                # infer missing positions given conditional for each axis
                x_missing = x_reorg[0] + mucondx
                y_missing = y_reorg[0] + mucondy
                z_missing = z_reorg[0] + mucondz
                std = np.sqrt(np.diag(sigcond))
                # final inferred coordinates and the error
                inf_coords = np.squeeze(np.concatenate((x_missing.T, y_missing.T, z_missing.T), axis=1))
                std_xyz = np.array([std, std, std])
                # replace missing coordinates with the inferred ones
                tr[missing_idxs,:] = inf_coords
                stack_imputed[n] = tr[:]

        stack_imputed = CoordsArrayStack(stack_imputed)
        
        return stack_imputed


####################



class PairwiseArrayStack(np.ndarray):

    # Initializing using __new__ instead of __init__ in line with Numpy
    def __new__(cls, arr:np.ndarray):
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1 or 2
    
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        s = arr.shape
        d = int(np.ceil(np.sqrt(s[-1] * 2)))
        if d * (d - 1) != s[-1] * 2:
            raise ValueError("Incompatible vector size. Must be a binomial coefficient" \
            "n choose 2 for some integer n >= 2.")
        
        obj = np.asarray(arr).view(cls)

        # add new attributes to the created instance
        obj.num_items = arr.shape[0]
        obj.num_bc = d

        return obj


    # Since new attributes were added in __new__
    def __array_finalize__(self, obj):
        # reset the attribute from passed original object
        self.num_items = getattr(obj, "num_items", None)
        self.num_bc = getattr(obj, "num_bc", None)


    def compute_ensemble(self, method="mean") -> np.ndarray:
        """Perform ensemble operations on the stack
        
        different methods: mean, median, std, sum, max, min
        """

        methods = {"mean": np.nanmean, "median": np.nanmedian, "std": np.nanstd,
                   "sum": np.nansum, "max": np.nanmax, "min": np.nanmin}
        assert method.strip().lower() in methods.keys()

        if self.num_items == 1:
            warnings.warn("Note: ensemble operation being performed on an array with only one item")

        return methods[method](self.data[:], axis=0)


    # NOTE jb's original function was different: data = 1 / (1 + (data / contact_threshold)**10)
    def compute_proximity(self, threshold:Number, hardness=1, keep_NaNs=False):
        """Compute hard or soft proximity using a threshold
        
        For hard proximity. Hardness describes sigmoid slope, 0 < hardness <= 1, where 1 means hard"""

        assert isinstance(threshold, Number)
        assert not ((self == 0) | (self == 1)).all()
        assert threshold <= np.nanmax(self)
        assert isinstance(hardness, Number) and 0 < hardness <= 1

        if self.shape[0] == 1:
            warnings.warn("Note: ensemble operation being performed on an array with only one item")

        # def _gt_lt_oper(X, t, b):
        #     """Switch between greater than or less than while performing a thresholding operation"""
        #     return X < t if b == "lt" else X > t
        
        mode = "hard" if hardness == 1 else "soft"

        if mode == "hard":
            if keep_NaNs:
                return np.where(np.isnan(self), np.nan, self <= threshold)
            return (self <= threshold).astype(int)
        
        # data = np.nan_to_num(data, nan=np.inf)  # convert NaN to Inf
        result = 1 / (1 + np.exp(hardness * (self - threshold)))

        if keep_NaNs:
            return result
        return np.nan_to_num(result, nan=0, copy=True)

    # TODO verify that the code works correctly, does not appear to do so right now
    def filter(self, efficiency=0.1) -> np.ndarray:
        """ Filter the dataset by minimum number of barcodes present

        Parameters
        ----------
        min_efficiency : minimum barcode detection efficiency

        Returns
        ----------
        filtered dataset
        """

        self_num_not_nan = np.sum(~np.isnan(self), axis=1)
        idxs = np.argwhere(self_num_not_nan > int(efficiency * self.shape[1]))

        return np.squeeze(self[idxs])
    

    # TODO finish the vectorised implementation for a 3D matrix, now it is only for 2D
    def reorganize(self, idxs) -> np.ndarray:      
        """ Reorganize a symmetric matrix with new positions for the previous indices

        Parameters
        ----------
        matrix : two-dimensional symmetric array
        idxs : indices using which `matrix` will be reorganised

        Returns
        ----------
        Reorganized matrix based on the new order of original indices
        """

        assert len(np.unique(idxs)) == len(idxs)
        assert np.all(np.logical_and(idxs >= 0, idxs < self.shape[0]))

        return self[:,idxs][idxs]


    def plot_pwds_avg(self, path_to_save:str, method="mean", form="triangle", cmap="Reds_r"):
        """ Plot the ensemble-average pairwise distance matrix
        
        Parameters
        --------
        path_to_save : str
            path to save the figure, format is taken automatically from the extension
        method : str
            mean or median (default: mean)
        form : str
            triangle or square representation (default: triangle)
        cmap : str
            colormap of the plot (default: Reds_r)
        """

        assert isinstance(path_to_save, str)
        assert form in ["triangle", "square"]

        matrix = self.compute_ensemble(method=method)
        if form == "triangle":
            visualization._plot_2d_triangle(matrix=matrix, path_to_save=path_to_save,
                                            cmap=cmap, label=f"{method} distance")
        else:
            visualization._plot_2d_square(matrix=matrix, path_to_save=path_to_save,
                                          cmap=cmap, label=f"{method} distance")


    def plot_proximity_avg(self, path_to_save:str, threshold:Number, hardness=1, form="triangle", cmap="Reds"):
        """ Plot the ensemble-average proximity matrix
        
        Parameters
        --------
        path_to_save : str
            path to save the figure, format is taken automatically from the extension
        form : str
            triangle or square representation (default: triangle)
        cmap : str
            colormap of the plot (default: Reds_r)
        """

        assert isinstance(path_to_save, str)
        assert form in ["triangle", "square"]

        proximity_stack = self.compute_proximity(threshold=threshold, hardness=hardness)
        matrix = np.nanmean(proximity_stack, axis=0)
        if form == "triangle":
            visualization._plot_2d_triangle(matrix=matrix, path_to_save=path_to_save, cmap=cmap, label="mean proximity")
        else:
            visualization._plot_2d_square(matrix=matrix, path_to_save=path_to_save, cmap=cmap, label="mean proximity")


####################