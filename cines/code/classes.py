from os import path, makedirs
from io import StringIO
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
from numbers import Number
import warnings
from sklearn.decomposition import LatentDirichletAllocation

####################


class CoordsArrayStack(np.ndarray):

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
        """ Compute the center of mass (CoM)"""

        return np.nanmean(self, axis=-2)


    def compute_pwds(self):
        """ Compute pairwise distances"""

        pwds = np.empty((self.num_items, int((self.num_bc * (self.num_bc - 1)) / 2), ))
        for i in range(self.num_items):
            pwds[i] = pdist(self[i])
        pwds = PairwiseArrayStack(pwds)
        
        return pwds


    # TODO more efficient approach for the naive method
    # TODO add geomedian and jost methods
    def impute(self, method="naive"):
        """ Missing barcode are replaced by NaN. If an isolated missing barcode is detected (meaning an isolated NaN value),
        the position of the missing barcode is computed as the mean position of the neighboring barcodes (before and after).

        @param trace: (numpy array) 2D array containing all the positions detected for a specific trace
        @param n_bc: (int) the total number of barcodes used for the experiment
        @return:
            corrected_trace : (numpy array) completed trace
            corrected_efficiency : (float) indicate the barcode efficiency of the newly corrected trace
            corrected_spot_ids : (numpy array) spot ids
        """

        method = method.lower()
        assert method in ["naive", "geomedian", "jost"]

        if ~ np.all(np.isnan(self)):
            warnings.warn("No NaNs to impute in the stack")
            return self
        
        if method == "naive":
            neighbor_means = (self[:, 2:] + self[:, :-2]) / 2   # compute the mean of all potential neighbors in order
            trace_imputed = self[:]
            is_nan = np.isnan(trace_imputed[:, 1:-1])
            trace_imputed[:, 1:-1][is_nan] = neighbor_means[np.where(is_nan)]   # replace NaNs with mean positions
        
        return CoordsArrayStack(trace_imputed)

    
    def compute_bc_effs(self):

        """ Compute barcode hybridization efficiency (percentage of barcode detection)"""

        bc_effs = np.zeros((self.num_bc,))
        for i in range(self.num_bc):
            num_bc_found = np.sum(~ np.isnan(self[:, i, :]).any(axis=1))
            bc_effs[i] = np.around(num_bc_found * 100 / self.num_items, decimals=1)

        return bc_effs


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


    def ensemble(self, method="mean") -> np.ndarray:
        """Perform ensemble operations using different methods: mean, median, std, sum, 
        max, min"""

        methods = {"mean": np.nanmean, "median": np.nanmedian, "std": np.nanstd,
                   "sum": np.nansum, "max": np.nanmax, "min": np.nanmin}
        assert method.strip().lower() in methods.keys()

        if self.shape[0] == 1:
            warnings.warn("Note: ensemble operation being performed on an array with only one item")

        return methods[method](self.data[:], axis=0)


    # def reorganize(self, new_idxs) -> np.ndarray:
    #     """Reorganize a symmetric matrix with new positions for the previous indices and
    #     return vector"""

    #     assert len(np.unique(new_idxs)) == len(new_idxs)
    #     assert np.all(np.logical_and(new_idxs >= 0, new_idxs < self.shape[0]))

    #     sq_data_reorg = squareform(self.data[:])[:,new_idxs][new_idxs]  # reorganize square symmetric maps
        
    #     return squareform(sq_data_reorg)    # reverting to vector form
    

    # TODO change the name of the function, binarize is no longer suitable
    # NOTE jb's function: data = 1 / (1 + (data / contact_threshold)**10)
    def binarize(self, threshold:Number, hardness=1, keep_NaNs=False, behavior="lt"):
        """ Binarize soft or hard using a threshold, behavior lt means all values less than threshold will be 1, 
        rest 0, gt for the opposite. Hardness describes sigmoid slope, 0 < hardness <= 10"""

        assert isinstance(threshold, Number)
        assert not ((self == 0) | (self == 1)).all()
        assert threshold <= np.nanmax(self)
        assert behavior == "lt" or "gt"
        assert isinstance(hardness, Number) and 0 <= hardness <= 1

        if self.shape[0] == 1:
            warnings.warn("Note: ensemble operation being performed on an array with only one item")

        def _oper(X, t, b):
            return X < t if b == "lt" else X > t
        
        mode = "hard" if hardness == 1 else "soft"

        if mode == "hard":
            if keep_NaNs:
                return np.where(np.isnan(self), np.nan, _oper(self, threshold, behavior))
            return _oper(self, threshold, behavior).astype(int)
        
        # data = np.nan_to_num(data, nan=np.inf)  # convert NaN to Inf
        slope = hardness if behavior == "lt" else -hardness
        result = 1 / (1 + np.exp(slope * (self - threshold)))

        if keep_NaNs:
            return result
        return np.nan_to_num(result, nan=0, copy=True)
    
    
    def filter(self, efficiency=0.1) -> np.ndarray:

        self_num_not_nan = np.sum(~np.isnan(self), axis=1)
        idxs = np.argwhere(self_num_not_nan > int(efficiency * self.shape[1]))

        return np.squeeze(self[idxs])


####################


class RawData:

    def __init__(self):
        self.fofdata = None
        self.unit = None
        self.assembly = None
        self.trace_ids = None
        self.traces = None
        self.num_traces = None
        self.num_bc = None
        self.bc_ids = None


    def __str__(self):
        return (f"FOF-CT data: {self.fofdata is not None}\n"
                f"Trace array: {self.traces is not None}\n"
                f"Genome assembly: {self.assembly}\n"
                f"Measurement unit: {self.unit}\n"
                f"Number of barcodes: {self.num_bc}\n"
                f"Number of traces: {self.num_traces}\n"
                )
    

    def _findnsplit(self, full_seq:str, sub_seq:str) -> list:
        """ Find the first occurrence of a string within another and split into two at a newline"""
        return full_seq[full_seq.index(sub_seq):].split("\n", maxsplit=1)
    

    def _extract_traces_from_fof(self):
        """Extract traces from FOF-CT Dataframe"""

        traces = np.empty((self.num_traces, self.num_bc, 3))
        traces[:] = np.nan
        # create the trace from the coordinates
        for j, curr_trace_id in enumerate(self.trace_ids):
            curr_data = self.fofdata[self.fofdata["Trace_ID"] == curr_trace_id]
            barcodes = [self.bc_ids[x] for x in curr_data["Chrom_Start"].to_list()]
            coords = curr_data[["X", "Y", "Z"]].to_numpy()
            traces[j, barcodes] = coords[:]
        return traces

    
    def load(self, path_to_file:str):
        """ Open a trace file; can either be a FOF-CT trace core file or a Numpy file. Former will be formatted as a pandas dataframe.
        @param file_path: (str) path to the file
        @return: data: (pd dataframe) contains the XYZ localizations and barcode positions for all localizations
        @return: unit: (str) indicate which unit was used for the data
        """

        assert path.isfile(path_to_file)

        if path_to_file.endswith(".npy"):
            self.traces = CoordsArrayStack(np.load(path_to_file, allow_pickle=True))
            assert self.traces.data.shape[-1] == 3
            del self.fofdata

        elif path_to_file.endswith(".csv"):

            # load the data and organize them as a dataframe
            contents = open(path_to_file, mode="r").read()  # load all contents as string
            table_header_raw, table_data_raw = self._findnsplit(contents, "##columns")  # split the header from the trace data
            colnames = table_header_raw[table_header_raw.index("(")+1:table_header_raw.index(")")].split(",")
            colnames = [c.strip() for c in colnames]  # get column names
            data = pd.read_csv(StringIO(table_data_raw), sep=",", header=None)  # get trace data
            data.columns = colnames  # assign column names to trace table

            # Locating other key metadata
            assembly = self._findnsplit(contents, "##genome_assembly")[0].split("=")[1]
            unit = self._findnsplit(contents, "##XYZ_unit")[0].split("=")[1]

            # Variable assignment
            self.fofdata = data
            self.unit = unit
            self.assembly = assembly

            self.trace_ids = np.unique(self.fofdata["Trace_ID"])
            self.num_traces = len(self.trace_ids)

            starts = np.unique(data["Chrom_Start"])
            self.bc_ids = dict(zip(starts, range(len(starts))))
            self.num_bc = len(starts)

            # Extraction of traces
            self.traces = CoordsArrayStack(self._extract_traces_from_fof())
        
        else:
            raise Exception("Invalid file extension. Only npy and csv files accepted")
    

    # def _compute_genomic_dist_map(self):
    #     """ Based on the genomic coordinates of the barcodes, compute a genomic distance map indicating the genomic
    #     separation between every pair of barcodes.

    #     @param lib: (pandas dataframe) coordinates of the barcodes
    #     @return: dist_map: (numpy array) genomic distance map in kbp
    #     """

    #     assert self.fofdata is not None
    #     mean_pos = ((self.fofdata["Chrom_End"] + self.fofdata["Chrom_Start"]) / 2).to_numpy(dtype=int)
    #     dist_map = pdist(mean_pos.reshape(-1, 1), metric="euclidean")
    #     dist_map = np.divide(dist_map, 1000)
    #     return dist_map


    # def _compute_3d_vs_genomic_distance(pairwise_dist, gen_dist, gen_dist_bins, dist_threshold=1., experiment=None,
    #                                 axis='X'):
    #     """ Using the genomic distance between the barcodes, compute the median 3d distance as a function of the genomic
    #     distance.

    #     @param pairwise_dist: (numpy array) pairwise distances for each individual trace
    #     @param gen_dist: (numpy array) 1D array indicating the genomic distance between all pairs of barcodes
    #     @param gen_dist_bins: (int) list of genomic distance used to build the histogram
    #     @param dist_threshold: (float) maximum distance above which the distance are considered unrealistic
    #     @param experiment: (int) indicate the number of the experiment that is being processed
    #     @param axis: (str) indicate which axis is being processed
    #     @return:
    #     dist: (pandas dataframe) contains all the data for the histogram

    #     """
    #     n_maps = pairwise_dist.shape[0]

    #     # compile all the distance map and aggregate genetic and euclidean distances
    #     merged_list = []
    #     for i in range(n_maps):
    #         merged = np.column_stack((gen_dist, pairwise_dist[i, :]))
    #         merged_list.append(merged)

    #     merged_dist = np.array(merged_list)
    #     merged_dist = np.reshape(merged_dist, (n_maps * len(gen_dist), 2))
    #     merged_dist[merged_dist[:, 1] > dist_threshold, 1] = np.nan

    #     # to make the plot easier to read, data are binned
    #     gen_bin = np.linspace(start=np.min(gen_dist), stop=np.max(gen_dist)+1, num=gen_dist_bins)
    #     d_vs_gen_dist = np.zeros((len(gen_bin), 3))
    #     for bin in range(len(gen_bin) - 1):
    #         idx = np.where((merged_dist[:, 0] >= gen_bin[bin]) & (merged_dist[:, 0] < gen_bin[bin+1]))
    #         d_vs_gen_dist[bin, :] = [gen_bin[bin], np.nanmedian(merged_dist[idx, 1]), len(idx[0])]

    #     # convert to dataframe
    #     dist = pd.DataFrame()
    #     dist['genomic distance (kbp)'] = d_vs_gen_dist[:, 0]
    #     dist['log10 genomic dist (kbp)'] = np.log10(d_vs_gen_dist[:, 0])
    #     dist['median euclidean distance (nm)'] = d_vs_gen_dist[:, 1]
    #     dist['log10 median dist (nm)'] = np.log10(d_vs_gen_dist[:, 1])
    #     dist['n_data'] = d_vs_gen_dist[:, 2]
    #     dist['experiment'] = [f'exp_{experiment}'] * len(gen_bin)
    #     dist['axis'] = [axis] * len(gen_bin)
    #     return dist


    # def _compute_inter_loci_genomic_dist(lib, n_bins):
    #     """ Compute the genomic distance separating two consecutive barcodes (from the experiment)

    #     @param lib: (pandas dataframe) coordinates of the barcodes
    #     @param n_bins: (int) total number of barcodes of the library
    #     @return: locus_dist: (pandas dataframe) contains the genomic distance in kbp
    #     """
    #     locus_dist = pd.DataFrame()
    #     mean_pos = (lib["Chrom_End"] + lib["Chrom_Start"]) / 2

    #     for bin in range(n_bins-1):
    #         d = mean_pos[bin + 1] - mean_pos[bin]
    #         d = int(d /1000)
    #         if bin == 0:
    #             d_interloci = [d]
    #             locus = [f'locus_{bin}_{bin+1}']
    #         else:
    #             d_interloci = np.concatenate((d_interloci, [d]), axis=0)
    #             locus = locus + [f'locus_{bin}_{bin+1}']

    #     locus_dist['locus'] = locus
    #     locus_dist['genomic distance (kbp)'] = d_interloci
    #     return locus_dist
    

    def process(self):
        """Process the trace data and return everything as a Processed data object"""

        assert self.traces is not None
        proc_data = ProcessedData()
        proc_data.traces = self.traces
        proc_data.bc_effs = self.traces.compute_bc_effs()
        proc_data.pwds = self.traces.compute_pwds()
        # proc_data.contacts = proc_data.pwds.binarize(threshold=contact_threshold)
        proc_data.bc_list = self.bc_ids
        proc_data.metadata = None
        return proc_data

    

    # def _compute_efficiency_map(pwd, root, tag_exp):
    #     """ Compute efficiency map.

    #     @param pwd: (numpy array) pairwise distance maps
    #     @param root: (str) root folder
    #     @param tag_exp: (str) indicate the type of experiment that is analyzed
    #     """
    #     n_maps = pwd.shape[0]
    #     efficiency = np.zeros((pwd.shape[1],))
    #     for n in range(pwd.shape[1]):
    #         efficiency[n] = np.sum(~np.isnan(pwd[:, n])) * 100 / n_maps
    #     efficiency = squareform(efficiency)

    #     plt.figure(figsize=(8, 6))
    #     plt.imshow(efficiency, cmap="hot")
    #     plt.colorbar(label="efficiency")
    #     plt.tight_layout()
    #     plt.title(f'{tag} n={pwd.shape[0]}')
    #     fig_name = f"Efficiency_{tag_exp}.png"
    #     fig_path = path.join(root, "barcode_efficiency", fig_name)
    #     plt.savefig(fig_path, format="png")
    #     plt.close()


    # def plot_pwd_maps(pwd, root, tag_exp):
    #     ensemble_pwdists = squareform(np.nanmedian(pwd, axis=0))

    #     plt.figure(figsize=(8, 6))
    #     plt.imshow(ensemble_pwdists, cmap="coolwarm")
    #     plt.colorbar(label="distance")
    #     plt.tight_layout()
    #     plt.title(f'{tag} n={pwd.shape[0]}')
    #     fig_name = f"{tag_exp}.png"
    #     fig_path = path.join(root, "maps", fig_name)
    #     plt.savefig(fig_path, format="png")
    #     plt.close()


    # def plot_contact_maps(pwd, thresh, root, tag_exp):
    #     contacts = np.where(np.isnan(pwd), np.nan, pwd < thresh)
    #     contacts_ensemble = squareform(np.nanmean(contacts, axis=0))
    #     np.fill_diagonal(contacts_ensemble, 1)

    #     plt.figure(figsize=(8, 6))
    #     plt.imshow(contacts_ensemble, cmap="Reds", vmin=0, vmax=0.6)
    #     plt.colorbar(label="contact")
    #     plt.tight_layout()
    #     plt.title(f'{tag} n={pwd.shape[0]}')
    #     fig_name = f"{tag_exp}.png"
    #     fig_path = path.join(root, "maps", fig_name)
    #     plt.savefig(fig_path, format="png")
    #     plt.close()


    # def plot_log_normalization_graph(dist_df, dim, saving_filename):
    #     """ Plot in logarithmic scale the distance versus the genomic distance for the raw data and after normalization shift.

    #     @param dist_df: (pandas dataframe) contains all the histogram distance data (Euclidean and genomic)
    #     @param dim: (dict) dictionary indicating the name of the axis
    #     @param saving_filename: (str) path where to save the figure
    #     """
    #     fig, axs = plt.subplots(nrows=3, ncols=1)
    #     for ax_name, ax in dim.items():
    #         df = dist_df.loc[dist_df["axis"] == ax_name, ['genomic distance (kbp)',
    #                                                     'median distance (nm)',
    #                                                     'experiment']]
    #         sns.scatterplot(data=df, x='genomic dist (kbp)', y='median dist (nm)',
    #                         hue="experiment", ax=axs[ax])
    #     fig.set_size_inches(8, 10)
    #     plt.savefig(saving_filename, dpi=100)
    #     plt.close(fig)



####################


class ProcessedData:

    def __init__(self):
        self.traces = None
        self.pwds = None
        self.bc_effs = None
        self.metadata = None
        self.bc_list = None
        self.contacts = None

    def __str__(self):
        return (f"traces: {self.traces is not None}\n"
                f"pwd: {self.pwds is not None}\n"
                f"bc_effs: {self.bc_effs is not None}\n"
                f"metadata: {self.metadata}\n"
                )


    def load(self, path_to_file:str):
        # assert path.isfile(path_to_file)
        assert path_to_file.endswith(".h5")
        with h5py.File(path_to_file, mode="r") as handler:
            datasets = {x:y[:] for x, y in handler.items()}
            attributes = {x:y[:] for x, y in handler.attrs.items()}
        self.traces = datasets.get("traces", None)
        self.pwds = datasets.get("pwds", None)
        self.bc_effs = datasets.get("bc_effs", None)
        self.metadata = attributes
        return 0


    def save(self, path_to_file:str, replace=True):
        if path.isfile(path_to_file) and not replace:
            raise FileExistsError("File already exists and replace is set to False")
        attr_labels = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        print(attr_labels)
        with h5py.File(path_to_file, mode="w") as handler:
            for al in attr_labels:
                attr = getattr(self, al)
                try:
                    print(al, type(attr), attr.dtype)
                    handler.create_dataset(name=al, data=attr)
                    print(f"Saved {al} successfully")
                except AttributeError:
                    pass
                except TypeError:
                    handler.attrs[al] = attr
                    print(f"Saved {al} successfully as attribute")
        return 0


    def viz_pwds_ensemble(self):
        pwds_ensemble = squareform(np.nanmean(self.pwds, axis=0))
        np.fill_diagonal(pwds_ensemble, 0)
        plt.figure(figsize=(8, 6))
        im = plt.imshow(pwds_ensemble, cmap="Reds_r")
        plt.colorbar(label="distance")
        plt.tight_layout()
        return im
    

    def viz_contacts_ensemble(self, threshold):
        contacts = np.where(np.isnan(self.pwds), np.nan, self.pwds < threshold)
        contacts_ensemble = squareform(np.nanmean(contacts, axis=0))
        np.fill_diagonal(contacts_ensemble, 1)
        vmin = np.nanmin(contacts_ensemble)
        vmax = np.quantile(contacts_ensemble, 0.99)
        if np.isnan(vmax):
            vmax = np.nanmax(contacts_ensemble)
        plt.figure(figsize=(8, 6))
        im = plt.imshow(contacts_ensemble, cmap="Reds", vmin=vmin, vmax=vmax)
        plt.colorbar(label="contact freq")
        plt.tight_layout()
        return im
    

####################


class DecompositionResult:

    def __init__(self, model:LatentDirichletAllocation, decomposition:np.ndarray):
        self.model = model
        self.decomposition = decomposition
        self.components = model.components_
            

    def save(self, path_to_dir:str):
        with h5py.File(path.join(path_to_dir, "results.h5"), mode="w") as handler:
            handler.create_dataset(name="components", data=self.components)
            handler.create_dataset(name="decomposition", data=self.decomposition)
        import joblib
        joblib.dump(self.model, path.join(path_to_dir, "model.jl"))


    # def plot_triangular_matrix(topic, ax=None, cmap="coolwarm"):
    #     """
    #     Plot a triangular matrix using imshow, and clip to a lower triangle.

    #     @param topic: (np.array) square matrix
    #     @param ax: matplotlib axis
    #     @param cmap: colormap string
    #     @return: imshow object
    #     """
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     import matplotlib.patches as patches
    #     from matplotlib import transforms

    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=(10, 10))

    #     # create a mask, selecting only the lower part of the topic array (using np.triu) and then apply it to the topic
    #     # array, selecting only the lower part of the array (np.ma.masked_array)
    #     n_bins = topic.shape[0]
    #     mask = np.triu(np.ones_like(topic, dtype=bool))
    #     topic_masked = np.ma.masked_array(topic, mask=~mask)

    #     # plot the masked array
    #     im = ax.imshow(topic_masked, origin='lower', cmap=cmap, interpolation='none')

    #     # rotate the plot by -45° to display the masked topic as a triangle with the diagonal as basis
    #     transform = transforms.Affine2D().rotate_deg(-45).scale(1 / np.sqrt(2), -1 / np.sqrt(2)) + ax.transData
    #     im.set_transform(transform)

    #     # Clip to triangle shape - in the new set of axis, define the coordinates of the three extremities of a triangle.
    #     # Note that we need to start at [-0.5, 0] instead of [0, 0] to avoid having a small portion of the triangle being
    #     # clipped
    #     coord = np.array([[-0.5, 0], [n_bins, 0], [n_bins / 2, n_bins / np.sqrt(2)]])
    #     triangle = patches.Polygon(coord, closed=True, transform=ax.transData)
    #     im.set_clip_path(triangle)

    #     # Adjust axes
    #     ax.set_xlim(-1, n_bins)
    #     ax.set_ylim(0, n_bins )
    #     ax.set_aspect('equal')
    #     ax.axis('off')

    #     return im