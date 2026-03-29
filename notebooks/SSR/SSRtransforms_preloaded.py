import re
import copy
import os
import random
import numpy as np
import scipy.io
from scipy.ndimage import uniform_filter, binary_closing, label
import torch
from sklearn.mixture import GaussianMixture
from skimage.exposure import match_histograms
from pathlib import Path
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.datasets import DatasetFolder
from typing import Any, Tuple

# Experiment 1: Custom DatasetFolder + Transformations for SSR augmentation.

####################################################################################
################### CUSTOM .MAT FILE LOADER ########################################
####################################################################################

def mat_file_loader(path):
    """
    Custom loader function for DatasetFolder.
    Replaces your open_mat_file().
    
    Args:
        path: Path to .mat file
    Returns:
        complex_data: Complex numpy array from .mat file
    """
    mat_contents = scipy.io.loadmat(path)
    return mat_contents['complex_img']

class DatasetFolderWithPath(DatasetFolder):
    """
    Minimal modification to DatasetFolder.
    
    Passes (sample, path) to transforms instead of just sample.
    This allows SSRAugmentation to look up GMM params by filepath.
    """
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns:
            tuple: (sample, target) where sample has been through transforms
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        
        # ONLY CHANGE: Pass (sample, path) to transform instead of just sample
        if self.transform is not None:
            sample = self.transform((sample, path))  # ← Pass filepath!
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target

#####################################################################################
################### TRANSFORMATIONS #################################################
#####################################################################################

#####################################################################################
################### MAGNITUDE #######################################################

class Magnitude:
    """
    Transform: extract magnitude.
    """
    def __call__(self, complex_data_or_tuple):
        """
        Args:
            complex_data: Complex numpy array from .mat file
        Returns:
            magnitude: Magnitude as numpy array
        """
        # Handle both: raw image OR (image, filepath) tuple
        if isinstance(complex_data_or_tuple, tuple):
            complex_data, filepath = complex_data_or_tuple
            pass_along = True
        else:
            complex_data = complex_data_or_tuple
            filepath = None
            pass_along = False
            
        magnitude = np.abs(complex_data)
        
        return (magnitude, filepath) if pass_along else magnitude

#####################################################################################
################### LOG MAPPING #####################################################

class LogMapping:
    """
    Transform: apply log mapping.
    """
    def __init__(self, c = 1000.0):
        self.c = c
    
    def __call__(self, magnitude_data_or_tuple):
        """
        Args:
            magnitude_data: Magnitude numpy array
        Returns:
            log_mapped: Log-mapped intensity [0, 1] as numpy array
        """
        # Handle both: raw image OR (image, filepath) tuple
        if isinstance(magnitude_data_or_tuple, tuple):
            magnitude, filepath = magnitude_data_or_tuple
            pass_along = True
        else:
            magnitude = magnitude_data_or_tuple
            filepath = None
            pass_along = False
        
        mag_min = magnitude.min()
        mag_max = magnitude.max()
        
        # Min-max normalization to [0, 1]
        if mag_max == mag_min:
            A = np.zeros_like(magnitude, dtype = np.float32)
        else:
            A = (magnitude - mag_min) / (mag_max - mag_min)
        
        # Apply log mapping formula
        I = np.log10(1 + (self.c * A)) / np.log10(1 + self.c)
        result = I.astype(np.float32)
        
        return (result, filepath) if pass_along else result

#######################################################################################
################ GAUSSIAN NOISE #######################################################

class GaussianNoise:
    """
    adds Gaussian noise to data
    """
    def __init__(self, mu = 0.0, sigma = 0.3):
        self.mu = mu
        self.sigma = sigma
    def __call__(self, numpy_arr_or_tuple):
        if isinstance(numpy_arr_or_tuple, tuple):
            numpy_array, filepath = numpy_arr_or_tuple
            pass_along = True
        else:
            numpy_array = numpy_arr_or_tuple
            pass_along = False
            
        noise = np.random.normal(loc = self.mu, scale = self.sigma, size = numpy_array.shape)
        I_noise = numpy_array + noise
        result = I_noise.astype(np.float32)
        
        # DO NOT clip here - values can go outside [0, 1]
        return (result, filepath) if pass_along else result

#######################################################################################
################# SSR #################################################################

def fit_gmm_for_image(processed_image):
    # Fit GMM (your existing code)
    image_flat = processed_image.flatten().reshape(-1, 1)
    gmm = GaussianMixture(n_components = 3, covariance_type = 'full', max_iter = 1000, n_init = 1)
    gmm.fit(image_flat)
    
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_
    sorted_idx = np.argsort(means)
    
    theta_C1 = {'mu': means[sorted_idx[0]], 'sigma': stds[sorted_idx[0]], 'pi': weights[sorted_idx[0]]}
    theta_C2 = {'mu': means[sorted_idx[1]], 'sigma': stds[sorted_idx[1]], 'pi': weights[sorted_idx[1]]}
    theta_T = {'mu': means[sorted_idx[2]], 'sigma': stds[sorted_idx[2]], 'pi': weights[sorted_idx[2]]}
    return {"theta_C1": theta_C1, "theta_C2": theta_C2, "theta_T": theta_T}

def build_gmm_cache(input_dir, processing_func = LogMapping(c = 1000.0)):
    
    input_path = Path(input_dir)
    mat_files = list(input_path.rglob("*.mat"))
    
    if len(mat_files) == 0:
        print(f"WARNING: No .mat files found in {input_dir}")
        return {}
    else:
        print(f"Found {len(mat_files)} .mat files")
    
    gmm_cache = {}

    for mat_file in tqdm(mat_files,  desc = "Fitting GMMs" ):
        complex_img = mat_file_loader(str(mat_file))
        magnitude_img = Magnitude()(complex_img)
        processed_img = processing_func(magnitude_img)
        
        gmm_paras = fit_gmm_for_image(processed_img)

        abs_path = str(mat_file.resolve())
        gmm_cache[abs_path] = gmm_paras

    return gmm_cache

class SSRAugmentation:
    """SSR augmentation - combines your GMM code with randomization."""
    def __init__(self, gmm_cache, alpha=0.6, beta=0.4, apply_prob=0.5, gaussian_noise = True, mu_s = 0.0, sigma_s = 0.3):
        self.gmm_cache = gmm_cache
        self.alpha = alpha
        self.beta = beta
        self.apply_prob = apply_prob
        self.gaussian_noise = gaussian_noise
        self.mu_s = mu_s
        self.sigma_s = sigma_s
    
    def __call__(self, img_or_tuple):
       
        # Handle tuple input from dataset
        if isinstance(img_or_tuple, tuple):
            processed_image, filepath = img_or_tuple
        else:
            # If not a tuple, can't use cache (e.g., validation without filepath)
            return img_or_tuple
        
        if np.random.random() >= self.apply_prob:
            return processed_image

        abs_path = str(Path(filepath).resolve())
        if abs_path not in self.gmm_cache:
            print(f"\nWARNING: {filepath} not in cache!")
            return processed_image
        
        gmm_params = self.gmm_cache[abs_path]
        theta_C1 = gmm_params['theta_C1']
        theta_C2 = gmm_params['theta_C2']
        theta_T = gmm_params['theta_T']
        
        # SSR Steps 2-3 (add gaussian noise, randomize, histogram match)
        
        if self.gaussian_noise:
            _noise = GaussianNoise(self.mu_s, self.sigma_s)
            I_noise = _noise(processed_image)
        else:
            I_noise = processed_image

        delta_mu = np.random.uniform(-self.alpha, self.alpha)
        delta_sigma = np.random.uniform(-self.beta, self.beta)
        
        theta_C1_new = {
            'mu': theta_C1['mu'] * (1 + delta_mu),
            'sigma': theta_C1['sigma'] * (1 + delta_sigma),
            'pi': theta_C1['pi']
        }
        theta_C2_new = {
            'mu': theta_C2['mu'] * (1 + delta_mu),
            'sigma': theta_C2['sigma'] * (1 + delta_sigma),
            'pi': theta_C2['pi']
        }
        
        n_pixels = processed_image.size
        n_C1 = int(theta_C1_new['pi'] * n_pixels)
        n_C2 = int(theta_C2_new['pi'] * n_pixels)
        n_T = n_pixels - n_C1 - n_C2
        
        samples_C1 = np.random.normal(theta_C1_new['mu'], theta_C1_new['sigma'], n_C1)
        samples_C2 = np.random.normal(theta_C2_new['mu'], theta_C2_new['sigma'], n_C2)
        samples_T = np.random.normal(theta_T['mu'], theta_T['sigma'], n_T)
        
        all_samples = np.concatenate([samples_C1, samples_C2, samples_T])
        np.random.shuffle(all_samples)
        I_sampled = all_samples.reshape(processed_image.shape)
        
        I_rand = match_histograms(I_noise, I_sampled)
        return np.clip(I_rand, 0.0, 1.0).astype(np.float32)

########################################################################################
################# NUMPY TO TENSOR ######################################################

class NumpyToTensor3Channel:
    """
    Transform: Convert numpy array to 3-channel PyTorch tensor.
    Equivalent to your npy_loader() logic.
    """
    def __call__(self, image):
        """
        Args:
            image: 2D numpy array (H, W)
        Returns:
            tensor: 3-channel tensor (3, H, W) for ResNet
        """
        if isinstance(image, tuple):
            image = image[0]
        # Convert to tensor
        tensor = torch.from_numpy(image)
        
        # Add channel dimension: (H, W) -> (1, H, W)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        
        # Convert to 3 channels: (1, H, W) -> (3, H, W)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        
        return tensor

###########################################################################################
################## FILTERING DATASET ######################################################
###########################################################################################

def extract_elev(path):
    match = re.search(r"elevDeg_(\d{3})", path)
    if match:
        return int(match.group(1))
    return None

def filter_by_elev(dataset, allowed_angles):
    filtered = []
    ds_cp = copy.deepcopy(dataset)
    for path, label in dataset.samples:
        elev = extract_elev(path)
        if elev in allowed_angles:
            filtered.append((path, label))
    ds_cp.samples = filtered
    ds_cp.imgs = filtered
    ds_cp.targets = [label for _, label in filtered]
    return ds_cp
###########################################################################################
################## Experiment 2: Choi segmenation #########################################
###########################################################################################

# read mstar dataset to get complex image
def read_mstar_clutter_complex(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()
    
    header_length = int(raw[:2000].decode('ascii', errors='ignore')
                        .split('PhoenixHeaderLength=')[1].split()[0])
    header = raw[:header_length].decode('ascii', errors='ignore')
    lines = header.splitlines()
    
    def get_field(name):
        return [l for l in lines if name in l][0].split('=')[1].strip()
    
    rows = int(get_field('NumberOfRows'))
    cols = int(get_field('NumberOfColumns'))
    native_header_length = int(get_field('native_header_length'))
    
    data_offset = header_length + native_header_length
    n_pixels = rows * cols
    
    data = np.frombuffer(raw[data_offset:], dtype='>u2')
    magnitude = data[:n_pixels].reshape(rows, cols).astype(np.float32)
    phase     = data[n_pixels:2*n_pixels].reshape(rows, cols).astype(np.float32)
    
    # reconstruct complex from polar form
    complex_data = magnitude * np.exp(1j * phase)
    
    return complex_data


# step 0: to get I_v
def readjust_intensity(image):
    return image / image.sum()

# step 1
def find_target_and_shadow_mask(image_step0):
    # target_step1 = np.where(image_step0 >= np.percentile(image_step0, 97), 1, 0)
    target_step1 = np.where(image_step0 >= np.percentile(image_step0, 85), 1, 0)
    # shadow_step1 = np.where(image_step0 <= np.percentile(image_step0, 25), 1, 0)
    shadow_step1 = np.where(image_step0 <= np.percentile(image_step0, 30), 1, 0)
    return target_step1, shadow_step1

# step 2
def counting_filter(mask, window_size = 5, threshold = 15):
    neighbor_count = uniform_filter(mask.astype(float), size = window_size) * (window_size ** 2)
    return np.where(neighbor_count >= threshold, 1, 0)

# step 3 
def structing_ele(shape = 5):
    ele = np.ones((shape, shape))
    ele[0, 0] = 0
    ele[0, -1] = 0
    ele[-1, 0] = 0
    ele[-1, -1] = 0
    return ele
def morphological_closing(mask, structuring_element = structing_ele(shape = 5)):
    return binary_closing(mask, structure = structuring_element).astype(int)

# step 4
def find_largest_connected_component(mask):
    labeled_array, num_features = label(mask)
    if num_features == 0:
        return np.zeros_like(mask)
    largest_component = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
    return (labeled_array == largest_component).astype(int)

# step 5
def mask_to_intensity(mask, image_step0):
    return mask * image_step0

def choi_segmentation(image):
    image_step0 = readjust_intensity(image)
    target_step1, shadow_step1 = find_target_and_shadow_mask(image_step0)
    target_step2, shadow_step2 = counting_filter(target_step1, window_size = 5, threshold = 15), counting_filter(shadow_step1, window_size = 5, threshold = 15)
    target_step3, shadow_step3 = morphological_closing(target_step2, structuring_element = structing_ele(shape = 5)), morphological_closing(shadow_step2, structuring_element = structing_ele(shape = 5))
    target_step4, shadow_step4 = find_largest_connected_component(target_step3), find_largest_connected_component(shadow_step3)
    clutter = 1 - (target_step4 + shadow_step4)
    # target_step5, shadow_step5 = mask_to_intensity(target_step4, image_step0), mask_to_intensity(shadow_step4, image_step0)
    return target_step4, shadow_step4, clutter

def random_clutter_crop(clutter_image, crop_size=128, 
                         zero_threshold=0.9, shadow_threshold=0.3,
                         max_attempts=100):
    H, W = clutter_image.shape
    for _ in range(max_attempts):
        r = np.random.randint(0, H - crop_size)
        c = np.random.randint(0, W - crop_size)
        crop = clutter_image[r:r+crop_size, c:c+crop_size]
        # reject dead zones
        if (crop > 0).mean() < zero_threshold:
            continue
        # reject shadow-dominated regions (too many low value pixels)
        if (crop < np.percentile(clutter_image, 25)).mean() > shadow_threshold:
            continue
        return crop, r, c
    raise ValueError("Could not find valid clutter crop")

def build_clutter_cache(meas_ds_w_path, clutter_dir, preloaded = None, crop_size=128):
    """
    For each measured image, preassign a random clutter crop location.
    Returns dict: {filepath: {clutter_file, row_start, col_start}}
    (row_end and col_end are just start + crop_size so no need to store them)
    """
    clutter_files = list(preloaded.keys()) if preloaded else [
        os.path.join(clutter_dir, f)
        for f in os.listdir(clutter_dir)
        if os.path.isfile(os.path.join(clutter_dir, f))
    ]

    clutter_cache = {}
    
    rand_clutter_ls = random.choices(clutter_files, k = len(meas_ds_w_path))

    for idx, file in enumerate(tqdm(rand_clutter_ls, desc = "Building clutter cache")):
        C_full_log = preloaded[file] if preloaded else LogMapping(c=1000.0)(Magnitude()(read_mstar_clutter_complex(file)))
        C, r, c = random_clutter_crop(C_full_log, 
                                crop_size = crop_size, 
                                zero_threshold = 0.9, 
                                shadow_threshold = 0.3, 
                                max_attempts = 100)
        abs_path = str(Path(meas_ds_w_path.samples[idx][0]).resolve())
        clutter_cache[abs_path] = (file, r, c)
    return clutter_cache

class Scenario2Merging:
    def __init__(self, clutter_cache, preloaded = None, crop_size = 128, c = 1000.0):
        self.clutter_cache = clutter_cache
        self.preloaded = preloaded
        self.crop_size = crop_size
        self.log_mapper = LogMapping(c = c)
    
    def __call__(self, img_or_tuple):
        if isinstance(img_or_tuple, tuple):
            I_meas, filepath = img_or_tuple
        else:
            return img_or_tuple

        abs_path = str(Path(filepath).resolve())
        if abs_path not in self.clutter_cache.keys():
            print(f"WARNING: {filepath} not in clutter cache!")
            return I_meas
        
        clutter_file, r, c = self.clutter_cache[abs_path]

        if self.preloaded and clutter_file in self.preloaded:
            clutter_log = self.preloaded[clutter_file]
        else:
            clutter_complex = read_mstar_clutter_complex(clutter_file)
            clutter_raw = Magnitude()(clutter_complex)
            clutter_log = self.log_mapper(clutter_raw)
        
        # crop — just array slicing, very fast
        C = clutter_log[r:r + self.crop_size, c:c + self.crop_size]
        
        # choi segmentation + merge
        M_target, M_shadow, M_clutter = choi_segmentation(I_meas)
        
        clutter_sum = M_clutter.sum()
        if clutter_sum == 0:
            return (I_meas, filepath)
        
        C_bar      = (C * M_clutter).sum() / clutter_sum
        I_meas_bar = (I_meas * M_clutter).sum() / clutter_sum
        d          = C_bar - I_meas_bar
        
        I_merged = (I_meas + d) * (M_target + M_shadow) + C * M_clutter
        I_merged = np.clip(I_merged, 0.0, 1.0).astype(np.float32)
        # I_merged = (I_merged).astype(np.float32)
        return (I_merged, filepath)