#!/usr/bin/env python

import h5py
import numpy as np
import argparse

# from mirror_check import calc_transmission

def h5_to_dict(h5file):
    def recursively_load_hdf5_group(group):
        ans = {}
        for key, item in group.items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = recursively_load_hdf5_group(item)
        return ans

    with h5py.File(h5file, 'r') as f:
        return recursively_load_hdf5_group(f)
    
    
def calc_transmission(pwr_a, pwr_b, g_a, g_b, logging=None):
    """
    Calculate the transmission, with uncertainty
    Parameters
    ---------
    pwr_a : Offset-subtracted power meter voltage, after the KB 
    pwr_b : Offset-subtracted power meter voltage, before the KB
    g_a : GMD after
    g_b : GMD before
    Returns
    ---------
    transmission, sigma
    """
    # dac_min_div = np.diff(np.sort(np.unique(pwr_a))).min()
    # dac_err_std = dac_min_div / np.sqrt(3)
    
    onset_a = np.where(pwr_a > pwr_a.max() /2)[0][0]
    onset_ga = int(np.round(onset_a * g_a.size / pwr_a.size))
    pwr_a = pwr_a[onset_a:]
    g_a = g_a[onset_ga:]
    
    onset_b = np.where(pwr_b > pwr_b.max() /2)[0][0]
    onset_gb = int(np.round(onset_b * g_b.size / pwr_b.size))
    pwr_b = pwr_b[onset_b:]
    g_b = g_b[onset_gb:]
    
    if logging is not None:
        msg = "Truncating off the data prior to the onset of beam power: "
        msg = msg + f"t<{onset_gb} (<{onset_ga}) for the mirror before (after)"
        if logging == "std":
            print(msg)
        else:
            logging.info(msg)
        
    pwr_a_mean = pwr_a.mean()
    pwr_b_mean = pwr_b.mean()
    g_a_mean = g_a.mean()
    g_b_mean = g_b.mean()
    transm = pwr_a_mean / pwr_b_mean * g_b_mean / g_a_mean
    relstderr = [x.std(ddof=1)/x.mean() for x in [pwr_a, pwr_b, g_a, g_b]]
    # for x in [pwr_a, pwr_b]:
    #     relstderr.append(dac_err_std/np.sqrt(x.size)/x.mean())
    relstderr = np.sum(np.array(relstderr)**2)**0.5
    return transm, transm * relstderr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read an HDF5 file and load its contents into a dictionary.')
    parser.add_argument('-f', '--hdf5_filename', type=str, help='Path to the HDF5 file')

    args = parser.parse_args()
    data = h5_to_dict(args.hdf5_filename)

    selected_data = {
            "bg_before" : data["background_noise"]["im3k4_power_meter_raw_voltage_buffer"],
            "bg_after"  : data["background_noise"]["im4k4_power_meter_raw_voltage_buffer"],
            "ppm_before": data["imager_before_mirror_data"]["im3k4_power_meter_raw_voltage_buffer"],
            "ppm_after" : data["imager_after_mirror_data"]["im4k4_power_meter_raw_voltage_buffer"],
            "gmd_before": data["imager_before_mirror_data"]["kfe_signals_kfe_gmd_avg_int"],
            "gmd_after" : data["imager_after_mirror_data"]["kfe_signals_kfe_gmd_avg_int"]
            }

    # data_median = {}
    # for k,v in selected_data.items():
    #     data_median[k] = np.median(v)
    ppm_after = selected_data["ppm_after"]
    ppm_before = selected_data["ppm_before"]
    bg_avg_noise_imager_after = selected_data['bg_after'].mean()
    bg_avg_noise_imager_before = selected_data['bg_before'].mean()
    gmd_avg_before = selected_data["gmd_before"]  
    gmd_avg_after = selected_data["gmd_after"]      # noqa: E501
    
    transmission, sgm_transmission = calc_transmission(ppm_after - bg_avg_noise_imager_after,
                                     ppm_before - bg_avg_noise_imager_before,
                                     gmd_avg_after,
                                     gmd_avg_before,
                                     "std")
    
    print(f"transmission = {transmission:.3f} +/- {sgm_transmission:.3f}")
    
    # transmission = ((data_median["ppm_after"] - data_median["bg_before"]) /
    #         (data_median["ppm_before"] - data_median["bg_after"])) *  +\
    # (data_median["gmd_before"] / data_median["gmd_after"])  # noqa: E501

    # print(f"transmission (median): {transmission}")
