#!/usr/bin/env python


"""
Mirror Transmission Check
========================

This script checks the transmission of a specified mirror and saves the
collected data into an HDF5 file. It performs the following steps:

1. Check attenuator transmission.
2. collect and save mirror motors positions.
3. measure background noise for all imagers.
4. Place the power meter before the mirror.
5. Collect data from the imager before the mirror for a specified duration.
6. Remove the power meter before the mirror.
7. Place the power meter after the mirror.
8. Collect data from the imager after the mirror for a specified duration.
9. Remove the power meter after the mirror.
10. Calculate deviation and check if it is larger than 0.05.
11. Calculate transmission and log the result.
12. Stores collected data into an HDF5 file.
13. Post scan results to the ELOG.
14. Save the transmission result to a PV.
"""

import datetime
import logging
import os
import sys
import time

import elog
import h5py
import numpy as np
from ophyd.signal import EpicsSignalRO
from pcdsdevices.mirror import KBOMirror
from pcdsdevices.pim import PPM

from tmo.mirrors import Mirror, MiscSignals
from tmo.mirrors import kfe_signals
from tmo.mirrors import mirrors

logging.basicConfig(level=logging.INFO)

from mirror_check_analysis import calc_transmission

class MirrorCheck:
    """
    MirrorCheck class for checking mirror transmission and saving data.

    Parameters:
    -----------
    mirror : Mirror
        The mirror object to be checked.
    output_path : str
        The path to save the collected data.
    scan_duration : float, optional
        The duration of each scan in seconds (default: 60.0).
    """
    def __init__(
        self,
        mirror: Mirror,
        output_path: str,
        scan_duration: float = 60.0,
        signals: MiscSignals = kfe_signals
    ):
        self._mirror = mirror
        self._output_path = output_path
        self._scan_duration = scan_duration
        self._signals = signals

    def check_attenuator(self, tolerance=1.0):
        """
        Check the attenuator transmission.

        Parameters:
        -----------
        tolerance : float, optional
            Tolerance level for transmission check (default: 1.0).

        Returns:
        --------
        Tuple[bool, float, float]
            A tuple containing a boolean indicating pass/fail, attenuator
            transmission, and tolerance.
        """
        attenuator_transmission = self._mirror.attenuator.get()
        if 1.0 - attenuator_transmission <= tolerance:
            return (True, attenuator_transmission, tolerance)
        else:
            return (False, attenuator_transmission, tolerance)

    def collect_data(self, signals, duration: float):
        """
        Collect data from a list of signals for a specified duration.

        Parameters:
        -----------
        signals : List[Signal]
            List of signals to collect data from.
        duration : float
            Duration of data collection in seconds.

        Returns:
        --------
        Dict[str, List]
            A dictionary containing collected data for each signal.
        """
        def _get_data(value, timestamp, **kwargs):
            obj = kwargs["obj"]
            if isinstance(value, np.ndarray):
                data[obj.name].extend(value)
            else:
                data[obj.name].append(value)

        data = {}
        uid = {}

        for signal in signals:
            data[signal.name] = []
            uid[signal.name] = signal.subscribe(_get_data)

        time.sleep(duration)

        for signal in signals:
            signal.unsubscribe(uid[signal.name])
            data[f"{signal.name}_mean"] = np.mean(data[signal.name])

        return data

    def collect_mirror_positions(self, mirror_obj: KBOMirror):
        """
        Collect and return positions of mirror motors.

        Parameters:
        -----------
        mirror_obj : KBOMirror
            The mirror object to extract motor positions from.

        Returns:
        --------
        Dict[str, float]
            A dictionary containing motor names and their respective positions.
        """
        positions = {}
        for keys in dir(mirror_obj):
            motor = getattr(mirror_obj, keys)
            if hasattr(motor, "position"):
                positions[motor.name] = getattr(motor, "position")
        return positions

    def save_dict_to_hdf5(self, file, data, group="/"):
        """
        Recursively save a dictionary to an HDF5 file.

        Parameters:
        -----------
        file : h5py.File
            The HDF5 file to save data to.
        data : Dict
            Data to be saved to the HDF5 file.
        group : str, optional
            The group in the HDF5 file to save data (default: "/").

        Returns:
        --------
        None

        """
        for key, item in data.items():
            if isinstance(item, dict):
                self.save_dict_to_hdf5(file, item, group + key + "/")
            else:
                file.create_dataset(group + key, data=item)

    def move_ppm(self, ppm: PPM, target: str, beam_rate_signal: EpicsSignalRO,
                 settling_time: float = 0.0):
        ppm.target(target, wait=True)
        while beam_rate_signal.get() < 10.0:
            logging.info("waiting for beam rate to go back ...")
            time.sleep(0.25)
        time.sleep(settling_time)

    def start_test(self):
        """
        Start the mirror transmission check process.

        The process involves multiple steps, including checking attenuator
        transmission, collecting data, calculating deviation and transmission,
        and saving data to an HDF5 file, posting results to the electronic
        logbook, and saving the result to PV.

        Returns:
        --------
        None
        """
        timestamp = time.time()
        dt = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

        logging.info("step #1: check attenuator transmission")
        status, attenuator_transmission, tolerance = self.check_attenuator()
        if status is False:
            logging.fatal(
                "check stopped: attenuator transmission \
                          {} is below tolerance {}".format(
                    attenuator_transmission, tolerance
                )
            )
            sys.exit()

        mirror_positions = {}
        logging.info("step #2: collect and save mirror motors positions")
        for mirror in self._mirror.mirror_objs:
            mirror_positions.update(self.collect_mirror_positions(mirror))

        logging.info("step #3: measure background noise for all imagers")
        self._signals.kfe_im2k0.target("DIAMOND", wait=True)
        ppms = [self._mirror.imager_before.power_meter.raw_voltage_buffer,
                self._mirror.imager_after.power_meter.raw_voltage_buffer]

        background_noise = self.collect_data(ppms, 60.0)

        self._signals.kfe_im2k0.target("OUT", wait=True)

        logging.info("step #4: Place power meter before mirror in")
        # self._mirror.imager_before.target("POWERMETER", wait=True)
        self.move_ppm(self._mirror.imager_before, "POWERMETER",
                      self._signals.kfe_rate, 10.0)

        logging.info(
            f"step #5: collect data from imager before mirror for +\
                     {self._scan_duration} seconds"
        )

        signals_before = [
            self._mirror.gmd.avg_int,
            self._mirror.gmd.mj,
            self._mirror.gmd.photons,
            self._mirror.gmd.transmission,

            self._mirror.xgmd.avg_int,
            self._mirror.xgmd.mj,
            self._mirror.xgmd.photons,
            self._mirror.xgmd.transmission,

            self._mirror.imager_before.power_meter.raw_voltage_buffer,
            self._mirror.imager_before.power_meter.calibrated_mj_buffer
        ]

        data_before = self.collect_data(signals_before,
                                        duration=self._scan_duration)

        logging.info("step #6: take power meter before mirror out")
        # self._mirror.imager_before.target("OUT", wait=True)
        self.move_ppm(self._mirror.imager_before, "OUT",
                      self._signals.kfe_rate)

        logging.info("step #7: Place power meter after mirror in")
        # self._mirror.imager_after.target("POWERMETER", wait=True)
        self.move_ppm(self._mirror.imager_after, "POWERMETER",
                      self._signals.kfe_rate, 10.0)

        logging.info(
            f"step #8: collect data from imager after  mirror for +\
                     {self._scan_duration} seconds"
        )

        signals_after = [
            self._mirror.gmd.avg_int,
            self._mirror.gmd.mj,
            self._mirror.gmd.photons,
            self._mirror.gmd.transmission,

            self._mirror.xgmd.avg_int,
            self._mirror.xgmd.mj,
            self._mirror.xgmd.photons,
            self._mirror.xgmd.transmission,

            self._mirror.imager_after.power_meter.raw_voltage_buffer,
            self._mirror.imager_after.power_meter.calibrated_mj_buffer
        ]

        data_after = self.collect_data(signals_after,
                                       duration=self._scan_duration)

        logging.info("step #9: take power meter after mirror out")
        # self._mirror.imager_after.target("OUT", wait=True)
        self.move_ppm(self._mirror.imager_after, "OUT", self._signals.kfe_rate)

        gmd_avg_before = data_before["kfe_signals_kfe_gmd_avg_int_mean"]    # noqa: E501
        gmd_avg_after = data_after["kfe_signals_kfe_gmd_avg_int_mean"]      # noqa: E501
        # ppm_before = data_before[
        #     f"{self._mirror.imager_before.name}_power_meter_raw_voltage_buffer_mean"        # noqa: E501
        # ]
        # ppm_after = data_after[
        #     f"{self._mirror.imager_after.name}_power_meter_raw_voltage_buffer_mean"         # noqa: E501
        # ]

        logging.info("step #10: calcualte deviation")
        deviation = 2.0 * (
            (gmd_avg_after - gmd_avg_after) / (gmd_avg_after + gmd_avg_after)
        )

        logging.info("deviation = {}".format(deviation))
        if deviation > 0.05:
            logging.fatal("beam fluction is larger than 0.05")
            sys.exit()

        logging.info("step #11: calcualte transmission")
        bg_avg_noise_imager_before = background_noise[f"{self._mirror.imager_before.name}_power_meter_raw_voltage_buffer_mean"]  # noqa: E501
        bg_avg_noise_imager_after = background_noise[f"{self._mirror.imager_after.name}_power_meter_raw_voltage_buffer_mean"]  # noqa: E501
        
        # Read the time-series, instead of the time-average,
        # to calculate the transmission
        ppm_after = data_after[
            f"{self._mirror.imager_after.name}_power_meter_raw_voltage_buffer"
        ]
        ppm_before = data_before[
            f"{self._mirror.imager_before.name}_power_meter_raw_voltage_buffer"
        ]
        gmd_avg_before = data_before["kfe_signals_kfe_gmd_avg_int"]  
        gmd_avg_after = data_after["kfe_signals_kfe_gmd_avg_int"]      # noqa: E501
        
        transmission, sgm_transmission = calc_transmission(ppm_after - bg_avg_noise_imager_after,
                                         ppm_before - bg_avg_noise_imager_before,
                                         gmd_avg_after,
                                         gmd_avg_before,
                                         logging)
        
        logging.info(f"transmission = {transmission:.3f} +/- {sgm_transmission:.3f}")

        data = {
            "datetime": dt,
            "timestamp": timestamp,
            "mirror_name": self._mirror.name,
            "mirror_positions": mirror_positions,
            "imager_before_mirror": self._mirror.imager_before.name,
            "imager_after_mirror": self._mirror.imager_after.name,
            "imager_before_mirror_data": data_before,
            "imager_after_mirror_data": data_after,
            "deviation": deviation,
            "transmission": transmission,
            "background_noise": background_noise
        }

        logging.info("step #12: store data into .h5 file")
        filepath = "{}/scans/{}".format(self._output_path, self._mirror.name)
        os.makedirs(filepath, exist_ok=True)
        hdf5_file_path = "{}/{}_{}.h5".format(
            filepath, self._mirror.name, data["datetime"]
        )
        with h5py.File(hdf5_file_path, "w") as hdf5_file:
            self.save_dict_to_hdf5(hdf5_file, data)

        logging.info("Data saved to {} successfully.".format(hdf5_file_path))

        logging.info("step #13: post scan results to elog")
        elog_obj = elog.HutchELog.from_registry()
        elog_obj.post(
            "{} transmission: {}. scan data path: {}".format(
                self._mirror.name, transmission, hdf5_file_path
            ),
            title="Mirror check",
            tags=[self._mirror.name],
        )

        logging.info("step #14: save result to PV")
        self._mirror.transmission.put(transmission)


def check_kbo1_transmission(output_path="/reg/g/pcds/pyps/apps/hutch-python/tmo/tmo/mirror-check", scan_duration=60.0):
    """
    perform transmission check for TMO KBO1 mirror using im3k4 and im4k4 power meters.
    Parameters
    ----------
    output_path : str, optional
        DESCRIPTION. The default is "/reg/g/pcds/pyps/apps/hutch-python/tmo/tmo/mirror-check".
    scan_duration : float, optional
        DESCRIPTION. The default is 60.0.
    Returns
    -------
    None.
    """

    kbo1 = MirrorCheck(mirrors["TMO_KBO1"], output_path, scan_duration)
    kbo1.start_test()


def check_kbo2_transmission(output_path="/reg/g/pcds/pyps/apps/hutch-python/tmo/tmo/mirror-check", scan_duration=60.0):
    """
    perform transmission check for TMO KBO2 mirror using im5k4 and im6k4 power meters.
    Parameters
    ----------
    output_path : str, optional
        DESCRIPTION. The default is "/reg/g/pcds/pyps/apps/hutch-python/tmo/tmo/mirror-check".
    scan_duration : float, optional
        DESCRIPTION. The default is 60.0.
    Returns
    -------
    None.
    """
    kbo2 = MirrorCheck(mirrors["TMO_KBO2"], output_path, scan_duration)
    kbo2.start_test()

