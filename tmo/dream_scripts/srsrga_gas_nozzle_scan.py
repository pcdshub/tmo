#!/usr/bin/python

from epics import Motor
from srsinst.rga import RGA300
import argparse
import datetime
import ipaddress
import json
import matplotlib.pyplot as plt
import numpy as np
import time


def valid_ip(value):
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid IP address")


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be a positive integer")
    return ivalue


def mass_range(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 300:
        raise argparse.ArgumentTypeError(f"Mass must be between 1 and 300")
    return ivalue


parser = argparse.ArgumentParser(description="Scan parameters")


class RgaScan:
    def __init__(self, ip, username='', password=''):
        self.ip = ip
        self.username = username
        self.password = password
        self.rga = self.connect()
        self.calibration_factor = 10E-16
        self.motor_x = Motor('DREAM:GSJN:MMS:X')
        self.motor_z = Motor('DREAM:GSJN:MMS:Z')

    def connect(self):
        return RGA300('tcpip', self.ip, self.username, self.password)

    def apply_settings(self):
        self.rga.scan.speed = 0
        self.rga.scan.resolution = 25
        # Set ionizer values
        self.rga.ionizer.electron_energy = 70
        self.rga.ionizer.ion_energy = 12
        self.rga.ionizer.focus_voltage = 90

        # Set the filament emsission current.
        self.rga.ionizer.emission_current = 1.0 	# in the unit of mA
        self.rga.filament.turn_on()	 # Turn on with the default emission cureent of 1 mA.

        self.rga.cem.voltage = self.rga.cem.stored_voltage
        self.rga.cem.turn_on() # turn electron multiplier on
        print("CEM warming for a minute...")
        time.sleep(60)

    def scan(self, mass, npoints):
        histogram = []
        self.apply_settings()
        for p in range(npoints):
            data_point = self.rga.scan.get_single_mass_scan(mass) * self.calibration_factor
            print("point #{:02d} | partial pressure: {:e}".format(p, data_point))
            histogram.append(data_point)

        self.rga.filament.turn_off()
        self.rga.cem.turn_off()

        plt.plot(histogram)
        plt.show()

        avg_pressure = np.mean(histogram)
        print("mass {} | average pressure: {:e} Torr".format(mass, avg_pressure))

    def gas_nozzle_scan(self, x_sps, z_sps, mass, npoints):
        self.apply_settings()
        data = {}
        data["datetime"] = str(datetime.datetime.now())
        data["x-axis"] = [str(p) for p in x_range]
        data["z-axis"] = [str(p) for p in z_range]
        data_points = []
        for x_sp in x_sps:
            for z_sp in z_sps:
                for _ in range(3):  # avoid backlash
                    self.motor_x.move(x_sp, wait=True)
                    self.motor_z.move(z_sp, wait=True)
                histogram = []
                for p in range(npoints):
                    data_point = self.rga.scan.get_single_mass_scan(mass) * self.calibration_factor
                    print("point #{:02d} | motor_x: {} |motor_z: {} |partial pressure: {:e}".format(p, x_sp, z_sp, data_point))
                    histogram.append(data_point)
                avg_pressure = np.mean(histogram)
                dp = {"motor_x_rbv": x_sp, "motor_z_rbv": z_sp, "avg_pressure": avg_pressure, "histogram": histogram}
                data_points.append(dp)
                print("-------------------------------------------------------")

        for _ in range(3):  # avoid backlash
            self.motor_x.move(x_range[0], wait=True)
            self.motor_z.move(z_range[0], wait=True)

        self.rga.filament.turn_off()
        self.rga.cem.turn_off()

        data["data_points"] = data_points
        data["mass"] = mass
        return data


if __name__ == '__main__':
    # connection parameters
    parser.add_argument('--ip', type=valid_ip, required=True, help="Target IP address (IPv4 or IPv6)")

    # X axis parameters
    parser.add_argument('--x_start', type=float, required=True, help="Start of x range")
    parser.add_argument('--x_end', type=float, required=True, help="End of x range")
    parser.add_argument('--x_points', type=positive_int, required=True, help="Number of points in x (must be > 0)")

    # Z axis parameters
    parser.add_argument('--z_start', type=float, required=True, help="Start of z range")
    parser.add_argument('--z_end', type=float, required=True, help="End of z range")
    parser.add_argument('--z_points', type=positive_int, required=True, help="Number of points in z (must be > 0)")

    # Mass and RGA parameters
    parser.add_argument('--mass', type=mass_range, required=True, help="Mass value (1 to 300)")
    parser.add_argument('--rga_points_per_amu', type=positive_int, required=True, help="Number of RGA points per amu (must be > 0)")

    args = parser.parse_args()

    chamber_rga = RgaScan(ip=args.ip)

    x_range = np.linspace(args.x_start, args.x_end, args.x_points)
    z_range = np.linspace(args.z_start, args.z_end, args.z_points)
    data = chamber_rga.gas_nozzle_scan(x_range, z_range, args.mass, args.rga_points_per_amu)
    json.dump(data, open("rga_scan_{}.json".format(datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")), 'w'), indent=4)
