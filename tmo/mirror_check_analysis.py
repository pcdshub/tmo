#!/usr/bin/env python

import h5py
import numpy as np
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read an HDF5 file and load its contents into a dictionary.')
    parser.add_argument('hdf5_filename', type=str, help='Path to the HDF5 file')

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

    data_median = {}
    for k,v in selected_data.items():
        data_median[k] = np.median(v)
    
    transmission = ((data_median["ppm_after"] - data_median["bg_before"]) /
            (data_median["ppm_before"] - data_median["bg_after"])) *  +\
    (data_median["gmd_before"] / data_median["gmd_after"])  # noqa: E501

    print(f"transmission (median): {transmission}")
