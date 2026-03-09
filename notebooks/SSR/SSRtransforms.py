import re
import copy
import numpy as np
import scipy.io
import torch
from sklearn.mixture import GaussianMixture
from skimage.exposure import match_histograms

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

#####################################################################################
################### TRANSFORMATIONS #################################################
#####################################################################################

#####################################################################################
################### LOG MAPPING #####################################################

class LogMapping:
    """
    Transform: Load .mat file complex data and apply log mapping.
    Combines your open_mat_file() and apply_log_mapping() functions.
    """
    def __init__(self, c=1000.0):
        self.c = c
    
    def __call__(self, complex_data):
        """
        Args:
            complex_data: Complex numpy array from .mat file
        Returns:
            log_mapped: Log-mapped intensity [0, 1] as numpy array
        """
        # Get magnitude
        magnitude = np.abs(complex_data)
        
        mag_min = magnitude.min()
        mag_max = magnitude.max()
        
        # Min-max normalization to [0, 1]
        if mag_max == mag_min:
            A = np.zeros_like(magnitude, dtype = np.float32)
        else:
            A = (magnitude - mag_min) / (mag_max - mag_min)
        
        # Apply log mapping formula
        I = np.log10(1 + self.c * A) / np.log10(1 + self.c)
        
        return I.astype(np.float32)

#######################################################################################
################ GAUSSIAN NOISE #######################################################

class GaussianNoise:
    """
    adds Gaussian noise to data
    """
    def __init__(self, mu = 0.0, sigma = 0.3):
        self.mu = mu
        self.sigma = sigma
    def __call__(self, numpy_array):
        noise = np.random.normal(loc = self.mu, scale = self.sigma, size = numpy_array.shape)
        I_noise = numpy_array + noise
    
        # DO NOT clip here - values can go outside [0, 1]
        return I_noise.astype(np.float32)

#######################################################################################
################# SSR #################################################################

class SSRAugmentation:
    """SSR augmentation - combines your GMM code with randomization."""
    def __init__(self, alpha=0.6, beta=0.4, apply_prob=0.5, gaussian_noise = True, mu_s = 0.0, sigma_s = 0.3):
        self.alpha = alpha
        self.beta = beta
        self.apply_prob = apply_prob
        self.gaussian_noise = gaussian_noise
        self.mu_s = mu_s
        self.sigma_s = sigma_s
    
    def __call__(self, processed_image):

        
        if np.random.random() >= self.apply_prob:
            return processed_image
        
        # Import your existing GMM function
        
        # Fit GMM (your existing code)
        image_flat = processed_image.flatten().reshape(-1, 1)
        gmm = GaussianMixture(n_components = 3, covariance_type = 'full', max_iter = 100, n_init = 50)
        gmm.fit(image_flat)
        
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_
        sorted_idx = np.argsort(means)
        
        theta_C1 = {'mu': means[sorted_idx[0]], 'sigma': stds[sorted_idx[0]], 'pi': weights[sorted_idx[0]]}
        theta_C2 = {'mu': means[sorted_idx[1]], 'sigma': stds[sorted_idx[1]], 'pi': weights[sorted_idx[1]]}
        theta_T = {'mu': means[sorted_idx[2]], 'sigma': stds[sorted_idx[2]], 'pi': weights[sorted_idx[2]]}
        
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
    ds_cp.targets = [label for _, label in filtered]
    return ds_cp