"""
Load the input maps for the correction and validation.
Currently some functions are duplicated for IDC. Later, the old functions should be removed.
"""
# pylint: disable=fixme, too-many-locals
# pylint: disable=consider-using-f-string
import random
import numpy as np
import scipy.constants

from tpcwithdnn.logger import get_logger

# Constants to calculate map file name based on its ordinal index.
SCALES_CONST = [0, 3, -3, 6, -6] # Indices of constant scaling of the mean maps
SCALES_LINEAR = [0, 3, -3] # Indices of linear scaling of the mean maps
SCALES_PARABOLIC = [0, 3, -3] # Indices of parabolic scaling of the mean maps

NELE_PER_ADC = 670 # A constant for charge-to-ADC (digitized value) normalization

ION_DRIFT_TIME_SIM = 200 # ion drift time (ms), corresponds to number of 1D IDCs to be used for FFT
NUM_FOURIER_COEFFS_MAX = 40

def get_mean_desc(mean_id):
    """
    Get map file name based on its ordinal index.

    :param int mean_id: index number of the mean map
    :return: mean file prefix
    :rtype: str
    """
    s_const = SCALES_CONST[mean_id // 9]
    s_lin = SCALES_LINEAR[(mean_id % 9) // 3]
    s_para = SCALES_PARABOLIC[mean_id % 3]
    return "%d-Const_%d_Lin_%d_Para_%d" % (mean_id, s_const, s_lin, s_para)


def load_data_original_idc(dirinput, event_index, z_range, use_rnd_augment):
    """
    The base function to load IDC data and filter it according to z_range.

    :param str dirinput: the directory with the input data
    :param list event_index: a list of [random_index, second_map_index] indices of the random
                             and the second reference map, respectively. The second map can be mean
                             or random, depending on use_rnd_augment.
    :param list z_range: a list of [min_z, max_z] values, the input and output data is taken
                         from this interval
    :param bool use_rnd_augment: if True, (random-random) map pairs are used,
                                 if False, (random-mean)
    :return: a vector of numpy arrays, one per each input data type:
             - r, rphi, z position
             - random and reference space charge
             - random and reference r, rphi and z distortion
             - random and reference r, rphi and z distortion correction
    :rtype: list
    """
    if use_rnd_augment:
        ref_prefix = "Random"
        ref_map_index = str(event_index[1])
    else:
        ref_prefix = "Mean"
        ref_map_index = get_mean_desc(event_index[1])

    files = ["%s/Pos/vecRPos.npy" % dirinput,
             "%s/Pos/vecPhiPos.npy" % dirinput,
             "%s/Pos/vecZPos.npy" % dirinput,
             "%s/%s/%s-vec%sSC.npy" % (dirinput, ref_prefix, ref_map_index, ref_prefix),
             "%s/Random/%d-vecRandomSC.npy" % (dirinput, event_index[0]),
             "%s/%s/%s-vec%sDistR.npy" % (dirinput, ref_prefix, ref_map_index, ref_prefix),
             "%s/Random/%d-vecRandomDistR.npy" % (dirinput, event_index[0]),
             "%s/%s/%s-vec%sDistRPhi.npy" % (dirinput, ref_prefix, ref_map_index, ref_prefix),
             "%s/Random/%d-vecRandomDistRPhi.npy" % (dirinput, event_index[0]),
             "%s/%s/%s-vec%sDistZ.npy" % (dirinput, ref_prefix, ref_map_index, ref_prefix),
             "%s/Random/%d-vecRandomDistZ.npy" % (dirinput, event_index[0]),
             "%s/%s/%s-vec%sCorrR.npy" % (dirinput, ref_prefix, ref_map_index, ref_prefix),
             "%s/Random/%d-vecRandomCorrR.npy" % (dirinput, event_index[0]),
             "%s/%s/%s-vec%sCorrRPhi.npy" % (dirinput, ref_prefix, ref_map_index, ref_prefix),
             "%s/Random/%d-vecRandomCorrRPhi.npy" % (dirinput, event_index[0]),
             "%s/%s/%s-vec%sCorrZ.npy" % (dirinput, ref_prefix, ref_map_index, ref_prefix),
             "%s/Random/%d-vecRandomCorrZ.npy" % (dirinput, event_index[0]),
             "%s/%s/%s-num%sZeroDIDCA.npy" % (dirinput, ref_prefix, ref_map_index, ref_prefix),
             "%s/%s/%s-num%sZeroDIDCC.npy" % (dirinput, ref_prefix, ref_map_index, ref_prefix),
             "%s/Random/%d-numRandomZeroDIDCA.npy" % (dirinput, event_index[0]),
             "%s/Random/%d-numRandomZeroDIDCC.npy" % (dirinput, event_index[0]),
             "%s/%s/%s-vec%sOneDIDCA.npy" % (dirinput, ref_prefix, ref_map_index, ref_prefix),
             "%s/%s/%s-vec%sOneDIDCC.npy" % (dirinput, ref_prefix, ref_map_index, ref_prefix),
             "%s/Random/%d-vecRandomOneDIDCA.npy" % (dirinput, event_index[0]),
             "%s/Random/%d-vecRandomOneDIDCC.npy" % (dirinput, event_index[0])]

    vec_z_pos_tmp = np.load(files[2])
    vec_sel_in_z = (z_range[0] <= vec_z_pos_tmp) & (vec_z_pos_tmp < z_range[1])

    mean_plus_prefix = "9-Const_3_Lin_0_Para_0"
    mean_minus_prefix = "18-Const_-3_Lin_0_Para_0"

    ref_mean_sc_plus_file = "%s/Mean/%s-vecMeanSC.npy" % (dirinput, mean_plus_prefix)
    ref_mean_sc_minus_file = "%s/Mean/%s-vecMeanSC.npy" % (dirinput, mean_minus_prefix)
    vec_der_ref_mean_sc = np.load(ref_mean_sc_plus_file)[vec_sel_in_z] - \
        np.load(ref_mean_sc_minus_file)[vec_sel_in_z]

    mat_der_ref_mean_corr = np.empty((3, vec_der_ref_mean_sc.size))
    ref_mean_corr_r_plus_file = "%s/Mean/%s-vecMeanCorrR.npy" % (dirinput, mean_plus_prefix)
    ref_mean_corr_r_minus_file = "%s/Mean/%s-vecMeanCorrR.npy" % (dirinput, mean_minus_prefix)
    mat_der_ref_mean_corr[0, :] = np.load(ref_mean_corr_r_plus_file)[vec_sel_in_z] \
        - np.load(ref_mean_corr_r_minus_file)[vec_sel_in_z]
    ref_mean_corr_rphi_plus_file = "%s/Mean/%s-vecMeanCorrRPhi.npy" % (dirinput, mean_plus_prefix)
    ref_mean_corr_rphi_minus_file = "%s/Mean/%s-vecMeanCorrRPhi.npy" % (dirinput, mean_minus_prefix)
    mat_der_ref_mean_corr[1, :] = np.load(ref_mean_corr_rphi_plus_file)[vec_sel_in_z] - \
        np.load(ref_mean_corr_rphi_minus_file)[vec_sel_in_z]
    ref_mean_corr_z_plus_file = "%s/Mean/%s-vecMeanCorrZ.npy" % (dirinput, mean_plus_prefix)
    ref_mean_corr_z_minus_file = "%s/Mean/%s-vecMeanCorrZ.npy" % (dirinput, mean_minus_prefix)
    mat_der_ref_mean_corr[2, :] = np.load(ref_mean_corr_z_plus_file)[vec_sel_in_z] \
        - np.load(ref_mean_corr_z_minus_file)[vec_sel_in_z]

    data = [np.load(f)[vec_sel_in_z] for f in files[:-8]] + \
        [vec_der_ref_mean_sc, mat_der_ref_mean_corr] + \
        [np.load(f) for f in files[-8:]]

    return data

def filter_idc_data(data_a, data_c, z_range):
    """
    Select the A-side and/or C-side data based on the z range.

    :param list data_a: list of arrays of values from the A-side
    :param list data_c: list of arrays of values from the C-side
    :param list z_range: a list of [min_z, max_z] values.
                         If the interval contains positive z, A-side data will be used.
                         Similarly, for any negative z C-side data is used.
    :return: tuple with selected data. If both A and C-side are selected,
             the correspondings arrays are stacked.
    :rtype: tuple
    """
    # TODO: Getter and application of Fourier coefficients need to be modified to handle both A and
    # C side at the same time
    if z_range[0] < 0 and z_range[1] > 0:  # pylint: disable=chained-comparison
        logger = get_logger()
        logger.fatal("Framework not yet fully prepared to use data from both A and C side at once.")

    output_data = []
    for data in data_a:
        output_data.append([])
    if z_range[1] > 0:
        for ind, data in enumerate(data_a):
            output_data[ind] = np.hstack((output_data[ind],
                                          data / (scipy.constants.e * NELE_PER_ADC))) # C -> ADC
    if z_range[0] < 0:
        for ind, data in enumerate(data_c):
            output_data[ind] = np.hstack((output_data[ind],
                                          data / (scipy.constants.e * NELE_PER_ADC))) # C -> ADC
    return tuple(output_data)

def load_data_original(dirinput, event_index):
    """
    Load the old input data.
    NOTE: Function for the old data, will be deprecated.

    :param str dirinput: the directory with the input data, value taken from the config file
    :param list event_index: a list of [random_index, mean_map_index] indices of the random
                             and the mean map, respectively.
    :return: list of vectors of r, rphi, z positions, mean and random space charge,
             and r, rphi, z mean and random distortions, unrestricted
    :rtype: list
    """

    ref_map_index = get_mean_desc(event_index[1])
    files = ["%s/Pos/vecRPos.npy" % dirinput,
             "%s/Pos/vecPhiPos.npy" % dirinput,
             "%s/Pos/vecZPos.npy" % dirinput,
             "%s/Mean/%s-vecMeanSC.npy" % (dirinput, ref_map_index),
             "%s/Random/%d-vecRandomSC.npy" % (dirinput, event_index[0]),
             "%s/Mean/%s-vecMeanDistR.npy" % (dirinput, ref_map_index),
             "%s/Random/%d-vecRandomDistR.npy" % (dirinput, event_index[0]),
             "%s/Mean/%s-vecMeanDistRPhi.npy" % (dirinput, ref_map_index),
             "%s/Random/%d-vecRandomDistRPhi.npy" % (dirinput, event_index[0]),
             "%s/Mean/%s-vecMeanDistZ.npy" % (dirinput, ref_map_index),
             "%s/Random/%d-vecRandomDistZ.npy" % (dirinput, event_index[0])]

    return [np.load(f) for f in files]

def load_data_derivatives_ref_mean(dirinput, z_range):
    """
    Load selected mean maps and calculate the derivatives of average space charge
    and average distortions.

    :param str dirinput: the directory with the input data, value taken from the config file
    :param list z_range: a list of [min_z, max_z] values, the input and output data will be
                         restricted to min_z <= z < max_z
    :return: tuple with a vector of SC derivative and a 2D array with derivatives of r, rphi and z
             distortions, respectively
    :rtype: tuple
    """
    z_pos_file = "%s/data/Pos/0-vecZPos.npy" % dirinput
    ref_mean_sc_plus_file = "%s/data/Mean/9-vecMeanSC.npy" % dirinput
    ref_mean_sc_minus_file = "%s/data/Mean/18-vecMeanSC.npy" % dirinput

    vec_z_pos = np.load(z_pos_file)
    vec_sel_z = (z_range[0] <= vec_z_pos) & (vec_z_pos < z_range[1])

    vec_der_ref_mean_sc = np.load(ref_mean_sc_plus_file)[vec_sel_z] - \
                          np.load(ref_mean_sc_minus_file)[vec_sel_z]

    mat_der_ref_mean_dist = np.empty((3, vec_der_ref_mean_sc.size))
    ref_mean_dist_r_plus_file = "%s/data/Mean/9-vecMeanDistR.npy" % dirinput
    ref_mean_dist_r_minus_file = "%s/data/Mean/18-vecMeanDistR.npy" % dirinput
    mat_der_ref_mean_dist[0, :] = np.load(ref_mean_dist_r_plus_file)[vec_sel_z] \
                                                - np.load(ref_mean_dist_r_minus_file)[vec_sel_z]
    ref_mean_dist_rphi_plus_file = "%s/data/Mean/9-vecMeanDistRPhi.npy" % dirinput
    ref_mean_dist_rphi_minus_file = "%s/data/Mean/18-vecMeanDistRPhi.npy" % dirinput
    mat_der_ref_mean_dist[1, :] = np.load(ref_mean_dist_rphi_plus_file)[vec_sel_z] - \
                                                np.load(ref_mean_dist_rphi_minus_file)[vec_sel_z]
    ref_mean_dist_z_plus_file = "%s/data/Mean/9-vecMeanDistZ.npy" % dirinput
    ref_mean_dist_z_minus_file = "%s/data/Mean/18-vecMeanDistZ.npy" % dirinput
    mat_der_ref_mean_dist[2, :] = np.load(ref_mean_dist_z_plus_file)[vec_sel_z] \
                                                - np.load(ref_mean_dist_z_minus_file)[vec_sel_z]

    return vec_der_ref_mean_sc, mat_der_ref_mean_dist

def mat_to_vec(opt_pred, mat_tuple):
    """
    Convert multidimensional arrays to flat vectors.

    :param list opt_pred: list of 3 binary values corresponding to activation of
                          r, rphi and z distortion corrections, taken from the config file
    :param tuple mat_tuple: tuple of arrays to be flattened
    :return: tuple of flattened input arrays
    :rtype: tuple
    """
    if sum(opt_pred) > 1:
        logger = get_logger()
        logger.fatal("Framework not yet fully prepared for more than one distortion direction.")

    sel_opts = np.array(opt_pred) > 0
    res = tuple(np.hstack(mat[sel_opts]) for mat in mat_tuple)
    return res

def downsample_data(data_size, downsample_npoints):
    """
    Downsample data - select randomly downsample_npoints voxels from the input data

    :param int data_size: size of the data to be downsampled
    :param int downsample_npoints: number of data voxels to be sampled
    :return: boolean vector that can be used as a mask for sampling 1D data
    :rtype: list
    """
    chosen = [False] * data_size
    for _ in range(downsample_npoints):
        sel_ind = random.randrange(0, data_size)
        while chosen[sel_ind]:
            sel_ind = random.randrange(0, data_size)
        chosen[sel_ind] = True
    return chosen


def get_fourier_coeffs(vec_oned_idc, num_fft_idcs,
                       num_fourier_coeffs_train, num_fourier_coeffs_apply):
    """
    Calculate Fourier transform and real and imaginary Fourier coefficients for a given vector.

    :param list vec_oned_idc: vector of 1D IDC values
    :param int num_fourier_coeffs_train: number of Fourier coefficients for training
    :param int num_fourier_coeffs_apply: number of Fourier coefficients for applying
    :return: numpy 1D array of interleaved real and imaginary Fourier coefficients
    :rtype: np.ndarray
    """
    diff_idcs = ION_DRIFT_TIME_SIM - num_fft_idcs
    if diff_idcs > 0:
        vec_oned_idc = vec_oned_idc[diff_idcs:ION_DRIFT_TIME_SIM]
    elif diff_idcs < 0:
        vec_oned_idc = np.append(np.random.choice(vec_oned_idc, size=abs(diff_idcs)), vec_oned_idc)
    dft = np.fft.fft(vec_oned_idc)
    dft_real = np.real(dft)[:num_fourier_coeffs_train]
    dft_imag = np.imag(dft)[:num_fourier_coeffs_train]
    num_fourier_coeffs_apply = min(num_fourier_coeffs_apply, num_fourier_coeffs_train)
    dft_real[num_fourier_coeffs_apply:num_fourier_coeffs_train] = 0.
    dft_imag[num_fourier_coeffs_apply:num_fourier_coeffs_train] = 0.
    return np.dstack((dft_real, dft_imag)).reshape(2 * num_fourier_coeffs_train)


def get_input_oned_idc_single_map(vec_r_pos, vec_phi_pos, vec_z_pos,
                                  mat_der_ref_mean_corr, dft_coeffs):
    """
    Create the input sample for 1D BDT correction for a single event map pair.

    :param np.ndarray vec_r_pos: vector of r positions
    :param np.ndarray vec_rphi_pos: vector of rphi positions
    :param np.ndarray vec_z_pos: vector of z positions
    :param np.ndarray vec_der_ref_mean_corr: vector of the derivative of average space charge
    :param np.ndarray dft_coeffs: vector of Fourier coefficients
    :return: an input sample (a vector) for 1D BDT correction
    :rtype: np.ndarray
    """
    inputs = np.zeros((vec_r_pos.size,
                       3 + mat_der_ref_mean_corr.shape[0] + dft_coeffs.size))
    for ind, pos in enumerate((vec_r_pos, vec_phi_pos, vec_z_pos)):
        inputs[:, ind] = pos
    inputs[:, 3:-dft_coeffs.size] = np.dstack(mat_der_ref_mean_corr)
    inputs[:, -dft_coeffs.size:] = dft_coeffs  # pylint: disable=invalid-unary-operand-type
    return inputs


def get_input_names_oned_idc(opt_usederivative, num_fourier_coeffs):
    """
    Get an array with names of the input parameters.

    :return: a list of names
    :rtype: list
    """
    input_names = ['r', 'phi', 'z']
    derivative_names = ['der_corr_r', 'der_corr_rphi', 'der_corr_z']
    for i_der, use_der in enumerate(opt_usederivative):
        if use_der == 1:
            input_names.append(derivative_names[i_der])
    for i in range(num_fourier_coeffs):
        input_names = input_names + ['c_real%d' % i, 'c_imag%d' % i]
    return input_names


def load_data_oned_idc(config, dirinput, event_index, downsample,
                       num_fft_idcs, num_fourier_coeffs_train, num_fourier_coeffs_apply):
    """
    Load inputs and outputs for one event for 1D IDC correction.

    :param CommonSettings config: a singleton settings object
    :param str dirinput: the directory with the input data
    :param list event_index: a list of [random_index, second_map_index] indices of the random
                             and the second reference map, respectively. The second map can be mean
                             or random, depending on use_rnd_augment.
    :param bool downsample: whether to downsample the data
    :param int num_fourier_coeffs_train: number of Fourier coefficients for training
    :param int num_fourier_coeffs_apply: number of Fourier coefficients for applying
    :return: tuple of inputs and expected outputs
    :rtype: tuple
    """
    [vec_r_pos, vec_phi_pos, vec_z_pos,
     *_,
     vec_mean_corr_r, vec_random_corr_r,
     vec_mean_corr_phi, vec_random_corr_phi,
     vec_mean_corr_z, vec_random_corr_z,
     _, mat_der_ref_mean_corr,
     _, _, _, _,
     vec_mean_oned_idc_a, vec_mean_oned_idc_c,
     vec_random_oned_idc_a, vec_random_oned_idc_c] = load_data_original_idc(dirinput, event_index,
                                                                            config.z_range,
                                                                            config.rnd_augment)

    # TODO: include also C side mean 0D IDC in case both sides to be used
    [*_,
     vec_ref_mean_corr_r, _,
     vec_ref_mean_corr_phi, _,
     vec_ref_mean_corr_z, _,
     _, _,
     num_ref_mean_zerod_idc_a, num_ref_mean_zerod_idc_c, _, _,
     _, _, _, _] = load_data_original_idc(dirinput, [0, 0], config.z_range, False)

    vec_oned_idc_fluc, num_ref_mean_zerod_idc = filter_idc_data(  # pylint: disable=unbalanced-tuple-unpacking
        (vec_random_oned_idc_a - vec_mean_oned_idc_a, num_ref_mean_zerod_idc_a),
        (vec_random_oned_idc_c - vec_mean_oned_idc_c, num_ref_mean_zerod_idc_c), config.z_range)
    dft_coeffs = get_fourier_coeffs(vec_oned_idc_fluc, num_fft_idcs, num_fourier_coeffs_train,
                                    num_fourier_coeffs_apply)

    vec_fluc_corr_r = vec_random_corr_r - vec_mean_corr_r
    vec_fluc_corr_phi = vec_random_corr_phi - vec_mean_corr_phi
    vec_fluc_corr_z = vec_random_corr_z - vec_mean_corr_z

    # TODO: this will not work properly if vec_exp_corr_fluc containes more than one
    # distortion direction
    vec_der_ref_mean_corr_r = mat_der_ref_mean_corr[0]
    vec_der_ref_mean_corr_rphi = mat_der_ref_mean_corr[1]
    vec_der_ref_mean_corr_z = mat_der_ref_mean_corr[2]
    if downsample:
        chosen_points = downsample_data(len(vec_z_pos), config.downsample_npoints)
        vec_r_pos = vec_r_pos[chosen_points]
        vec_phi_pos = vec_phi_pos[chosen_points]
        vec_z_pos = vec_z_pos[chosen_points]
        vec_ref_mean_corr_r = vec_ref_mean_corr_r[chosen_points]
        vec_ref_mean_corr_phi = vec_ref_mean_corr_phi[chosen_points]
        vec_ref_mean_corr_z = vec_ref_mean_corr_z[chosen_points]
        vec_fluc_corr_r = vec_fluc_corr_r[chosen_points]
        vec_fluc_corr_phi = vec_fluc_corr_phi[chosen_points]
        vec_fluc_corr_z = vec_fluc_corr_z[chosen_points]
        vec_der_ref_mean_corr_r = vec_der_ref_mean_corr_r[chosen_points]
        vec_der_ref_mean_corr_rphi = vec_der_ref_mean_corr_rphi[chosen_points]
        vec_der_ref_mean_corr_z = vec_der_ref_mean_corr_z[chosen_points]

    mat_der_ref_mean_corr_sel = np.array([vec_der_ref_mean_corr_r,
                                          vec_der_ref_mean_corr_rphi,
                                          vec_der_ref_mean_corr_z])
    mat_der_ref_mean_corr_sel = mat_der_ref_mean_corr_sel[np.array(config.opt_usederivative) > 0]

    inputs = get_input_oned_idc_single_map(vec_r_pos, vec_phi_pos, vec_z_pos,
                                           mat_der_ref_mean_corr_sel, dft_coeffs)

    mat_fluc_corr = np.array([vec_fluc_corr_r,
                              vec_fluc_corr_phi,
                              vec_fluc_corr_z])
    mat_fluc_corr = np.dstack(mat_fluc_corr[np.array(config.opt_predout) > 0])[0]

    vec_ref_mean_zerod_idc = np.full(vec_ref_mean_corr_r.shape[0],
                                     num_ref_mean_zerod_idc.astype('float32'))
    mat_ref_mean_values = np.dstack(np.array([vec_ref_mean_corr_r, vec_ref_mean_corr_phi,
                                              vec_ref_mean_corr_z, vec_ref_mean_zerod_idc]))[0]

    return inputs, mat_fluc_corr, mat_ref_mean_values


def load_data(dirinput, event_index, z_range):
    """
    Load files for specified event_index pair from the input directory, restricted to the z_range.
    NOTE: Function for the old data, will be deprecated.

    :param str dirinput: the directory with the input data, value taken from the config file
    :param list event_index: a list of [random_index, second_map_index] indices of the random
                             and the second reference map, respectively.
                             The second map can be mean or random, depending on the settings.
    :param list z_range: a list of [min_z, max_z] values, the input and output data will be
                         restricted to min_z <= z < max_z
    :return: list of vectors of mean space charge, space-charge fluctuations
             and r, rphi, z distortion fluctuations, restricted to the z_range
    :rtype: list
    """

    [_, _, vec_z_pos,
     vec_mean_sc, vec_random_sc,
     vec_mean_dist_r, vec_random_dist_r,
     vec_mean_dist_rphi, vec_random_dist_rphi,
     vec_mean_dist_z, vec_random_dist_z] = load_data_original(dirinput, event_index)

    vec_sel_in_z = (z_range[0] <= vec_z_pos) & (vec_z_pos < z_range[1])

    vec_mean_sc = vec_mean_sc[vec_sel_in_z]
    vec_fluctuation_sc = vec_random_sc[vec_sel_in_z] - vec_mean_sc

    vec_fluctuation_dist_r = vec_random_dist_r[vec_sel_in_z] - vec_mean_dist_r[vec_sel_in_z]
    vec_fluctuation_dist_rphi = vec_random_dist_rphi[vec_sel_in_z] -\
        vec_mean_dist_rphi[vec_sel_in_z]
    vec_fluctuation_dist_z = vec_random_dist_z[vec_sel_in_z] - vec_mean_dist_z[vec_sel_in_z]

    return [vec_mean_sc, vec_fluctuation_sc, vec_fluctuation_dist_r,
            vec_fluctuation_dist_rphi, vec_fluctuation_dist_z]


def load_train_apply(dirinput, event_index, z_range,
                     grid_r, grid_rphi, grid_z, opt_train, opt_pred):
    """
    Load inputs and outputs for training / apply for one event.
    NOTE: Function for the old data, will be deprecated.

    :param str dirinput: the directory with the input data, value taken from the config file
    :param list event_index: a list of [random_index, mean_map_index] indices of the random
                             and the mean map, respectively.
    :param list z_range: a list of [min_z, max_z] values, the input and output data is taken
                         from this interval
    :param int grid_r: grid granularity (number of voxels) along r-axis
    :param int grid_rphi: grid granularity (number of voxels) along rphi-axis
    :param int grid_z: grid granularity (number of voxels) along z-axis
    :param list opt_train: list of 2 binary values corresponding to activating the train input of
                           average space charge and space-charge fluctuations, respectively,
                           taken from the config file
    :param list opt_pred: list of 3 binary values corresponding to activating the prediction of
                          r, rphi and z distortion corrections, taken from the config file
    :return: tuple of inputs and expected outputs
    :rtype: tuple
    """
    [vec_mean_sc, vec_fluctuation_sc, vec_fluctuation_dist_r,
     vec_fluctuation_dist_rphi, vec_fluctuation_dist_z] = \
        load_data(dirinput, event_index, z_range)
    dim_input = sum(opt_train)
    dim_output = sum(opt_pred)
    inputs = np.empty((grid_rphi, grid_r, grid_z, dim_input))
    exp_outputs = np.empty((grid_rphi, grid_r, grid_z, dim_output))

    indexfillx = 0
    if opt_train[0] == 1:
        inputs[:, :, :, indexfillx] = \
                vec_mean_sc.reshape(grid_rphi, grid_r, grid_z)
        indexfillx = indexfillx + 1
    if opt_train[1] == 1:
        inputs[:, :, :, indexfillx] = \
                vec_fluctuation_sc.reshape(grid_rphi, grid_r, grid_z)

    if dim_output > 1:
        logger = get_logger()
        logger.fatal("YOU CAN PREDICT ONLY 1 DISTORTION. The sum of opt_predout == 1")

    flucs = np.array((vec_fluctuation_dist_r, vec_fluctuation_dist_rphi, vec_fluctuation_dist_z))
    sel_flucs = flucs[np.array(opt_pred) == 1]
    for ind, vec_fluctuation_dist in enumerate(sel_flucs):
        exp_outputs[:, :, :, ind] = \
                vec_fluctuation_dist.reshape(grid_rphi, grid_r, grid_z)

    return inputs, exp_outputs


def get_event_mean_indices(range_rnd_index_train, range_mean_index, ranges, use_rnd_augment):
    """
    Select randomly event pair indices for train / validation / apply.

    :param int range_rnd_index_train: number of random event maps available
    :param int range_mean_index: number of mean event maps available
    :param dict ranges: dictionary of lists of event ranges for train / validation / apply
    :param bool use_rnd_augment: if True, (random-random) map pairs are used,
                                 if False, (random-mean)
    :return: list of all selected map indices and dictionary with selected
             train / validation / apply indices
    :rtype: tuple(list, dict)
    """
    all_indices_events_means = []
    range_ref_index = range_mean_index
    if use_rnd_augment:
        range_ref_index = range_rnd_index_train
    for ievent in np.arange(range_rnd_index_train[0], range_rnd_index_train[1] + 1):
        for iref in np.arange(range_ref_index[0], range_ref_index[1] + 1):
            if use_rnd_augment and ievent <= iref:
                continue
            all_indices_events_means.append([ievent, iref])
    sel_indices_events_means = random.sample(all_indices_events_means, ranges["apply"][1] + 1)

    indices_train = sel_indices_events_means[ranges["train"][0]:ranges["train"][1]]
    indices_val = sel_indices_events_means[ranges["validation"][0]:ranges["validation"][1]]
    indices_apply = sel_indices_events_means[ranges["apply"][0]:ranges["apply"][1]]

    partition = {"train": indices_train,
                 "validation": indices_val,
                 "apply": indices_apply}

    return sel_indices_events_means, partition


# ======================================================================
# [BARU] Jalur struktural TDL+PINN: pemuat grid + create-data sintetik
# PURPOSE  : (1) load_event_grid: baca pasangan (Random, Mean) format npy
#            ASLI TPCwithDNN ke grid (nphi,nr,nz) via vektor Pos (robust
#            terhadap urutan); input = fluktuasi SC, label = fluktuasi
#            distorsi. (2) synthetic_event: gumpalan acak v1 (smoke).
#            (3) synthetic_event_mc: MC statistik ala Pers.1 paper ALICE -
#            fluktuasi = peta_acak(N ~ Poisson(n_pileup) deposit berbobot
#            fluk multiplisitas x muatan, posisi radial ~ 1/r^2) dikurangi
#            MEAN EMPIRIS (rata-rata M peta, di-cache) - meniru definisi
#            fluktuasi (random - mean) pipeline ALICE.
# CATATAN  : MC STATISTIK, BUKAN MC detektor penuh (GEANT/O2).
# STATUS   : kalibrasi satuan rho vs O2 masih PLACEHOLDER.
# DIPANGGIL: dnn_optimiser (train/apply).
# ======================================================================

def load_positions_grid(dirinput):
    """SUBROUTINE load_positions_grid | PURPOSE: sumbu unik + pemetaan
    flat->(iphi,ir,iz) dari vecPos via searchsorted."""
    rp = np.load("%s/Pos/vecRPos.npy" % dirinput)
    pp = np.load("%s/Pos/vecPhiPos.npy" % dirinput)
    zp = np.load("%s/Pos/vecZPos.npy" % dirinput)
    r_u, p_u, z_u = np.unique(rp), np.unique(pp), np.unique(zp)
    idx = (np.searchsorted(p_u, pp), np.searchsorted(r_u, rp),
           np.searchsorted(z_u, zp))
    return (p_u, r_u, z_u), idx


def _to_grid(vec, idx, shape):
    g = np.zeros(shape, dtype=np.float64)
    ip, ir, iz = idx
    g[ip, ir, iz] = vec
    return g


def load_event_grid(dirinput, rnd_index, mean_id=0):
    """SUBROUTINE load_event_grid | PURPOSE: satu event data riil:
    (axes, rho_fluc_grid, [dDistR, dDistRPhi, dDistZ] grid)."""
    axes, idx = load_positions_grid(dirinput)
    shape = tuple(len(a) for a in axes)
    mdesc = get_mean_desc(mean_id)
    sc = np.load("%s/Random/%d-vecRandomSC.npy" % (dirinput, rnd_index)) \
        - np.load("%s/Mean/%s-vecMeanSC.npy" % (dirinput, mdesc))
    labels = []
    for v in ("DistR", "DistRPhi", "DistZ"):
        d = np.load("%s/Random/%d-vecRandom%s.npy" % (dirinput, rnd_index, v)) \
            - np.load("%s/Mean/%s-vecMean%s.npy" % (dirinput, mdesc, v))
        labels.append(_to_grid(d, idx, shape))
    return axes, _to_grid(sc, idx, shape), labels


def synthetic_event(cx, seed=0, n_blob=6, amp=1.0):
    """SUBROUTINE synthetic_event | PURPOSE: gumpalan Gaussian acak v1
    (mode 'blobs') untuk smoke-test cepat."""
    rng = np.random.default_rng(seed)
    P, R, Z = cx.grid_shape()
    ip, ir, iz = np.meshgrid(np.arange(P), np.arange(R), np.arange(Z),
                             indexing="ij")
    r, z, ph = cx.r[ir], cx.z[iz], cx.phi[ip]
    rho = np.zeros((P, R, Z))
    for _ in range(n_blob):
        r0 = rng.uniform(cx.r[2], cx.r[-3]); z0 = rng.uniform(cx.z[2], cx.z[-3])
        sr = rng.uniform(2, 5) * cx.dr;      sz = rng.uniform(2, 5) * cx.dz
        m = rng.integers(0, 4);              a = rng.normal(0, amp)
        rho += a * np.exp(-((r - r0) ** 2) / (2 * sr ** 2)
                          - ((z - z0) ** 2) / (2 * sz ** 2)) \
            * np.cos(m * ph + rng.uniform(0, 2 * np.pi))
    return rho


def _mc_single_map(cx, rng, n_pileup, mult_rel_sigma, charge_rel_sigma):
    """SUBROUTINE _mc_single_map | PURPOSE: satu peta 'random' MC: N ~
    Poisson(n_pileup) deposit Gaussian, bobot fluk multiplisitas x muatan,
    posisi radial ~ 1/r^2 (inverse-CDF), phi & z seragam."""
    P, R, Z = cx.grid_shape()
    ip, ir, iz = np.meshgrid(np.arange(P), np.arange(R), np.arange(Z),
                             indexing="ij")
    r, z, ph = cx.r[ir], cx.z[iz], cx.phi[ip]
    n_ev = rng.poisson(n_pileup)
    w = np.clip(1.0 + mult_rel_sigma * rng.standard_normal(n_ev), 0.05, None) \
        * np.clip(1.0 + charge_rel_sigma * rng.standard_normal(n_ev),
                  0.05, None)
    u = rng.uniform(0, 1, n_ev)
    inv_rmin, inv_rmax = 1.0 / cx.r[0], 1.0 / cx.r[-1]
    r0 = 1.0 / (inv_rmin - u * (inv_rmin - inv_rmax))
    z0 = rng.uniform(cx.z[0], cx.z[-1], n_ev)
    p0 = rng.uniform(0, 2 * np.pi, n_ev)
    sr, sz, sp = 1.5 * cx.dr, 1.5 * cx.dz, 1.5 * cx.dphi
    total = np.zeros((P, R, Z))
    for i in range(n_ev):
        dphi_w = ((ph - p0[i] + np.pi) % (2 * np.pi)) - np.pi
        total += w[i] * np.exp(-((r - r0[i]) ** 2) / (2 * sr ** 2)
                               - ((z - z0[i]) ** 2) / (2 * sz ** 2)
                               - (dphi_w ** 2) / (2 * sp ** 2))
    return total


_MC_MEAN_CACHE = {}


def synthetic_event_mc(cx, seed=0, n_pileup=3000, mult_rel_sigma=0.3,
                       charge_rel_sigma=0.5, n_mean_maps=6):
    """SUBROUTINE synthetic_event_mc
    PURPOSE  : fluktuasi MC statistik ala Pers.1 ALICE: peta_random(seed)
               dikurangi MEAN EMPIRIS (rata-rata n_mean_maps peta, seed
               tetap, DI-CACHE per konfigurasi) - meniru definisi
               fluktuasi (random - mean) pipeline TPCwithDNN. Skala
               relatif ~ 1/sqrt(n_pileup) * sqrt(1+var_mult+var_charge).
    DIPANGGIL: dnn_optimiser (synthetic_mode: mc)."""
    key = (cx.grid_shape(), n_pileup, mult_rel_sigma, charge_rel_sigma,
           n_mean_maps)
    if key not in _MC_MEAN_CACHE:
        rng_m = np.random.default_rng(987654321)
        acc = np.zeros(cx.grid_shape())
        for _ in range(n_mean_maps):
            acc += _mc_single_map(cx, rng_m, n_pileup,
                                  mult_rel_sigma, charge_rel_sigma)
        _MC_MEAN_CACHE[key] = acc / n_mean_maps
    mean_map = _MC_MEAN_CACHE[key]
    rng = np.random.default_rng(seed)
    rnd = _mc_single_map(cx, rng, n_pileup, mult_rel_sigma, charge_rel_sigma)
    return (rnd - mean_map) / (np.abs(mean_map).mean() + 1e-30)
