from __future__ import annotations

from ophyd import Device, Component as Cpt
from ophyd.utils.epics_pvs import _wait_for_value
from ophyd.signal import EpicsSignal, EpicsSignalRO
from enum import IntEnum

import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AtomicMass(IntEnum):
    """
    Integer-based Enum of chemical elements with approximate atomic mass
    (nearest integer), used for simplified m/z calculations in mass spectrometry.

    This class maps each chemical element symbol to its approximate
    atomic mass rounded to the nearest integer.

    Examples
    --------
    >>> AtomicMassEnum.C
    <AtomicMassEnum.C: 12>

    >>> AtomicMassEnum['Fe'].value
    56

    >>> [e.name for e in AtomicMassEnum if e.value < 20]
    ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F']
    """

    H = 1     #: Hydrogen
    He = 4    #: Helium
    Li = 7    #: Lithium
    Be = 9    #: Beryllium
    B = 11    #: Boron
    C = 12    #: Carbon
    N = 14    #: Nitrogen
    O = 16    #: Oxygen  # noqa: E741
    F = 19    #: Fluorine
    Ne = 20   #: Neon
    Na = 23   #: Sodium
    Mg = 24   #: Magnesium
    Al = 27   #: Aluminum
    Si = 28   #: Silicon
    P = 31    #: Phosphorus
    S = 32    #: Sulfur
    Cl = 35   #: Chlorine
    K = 39    #: Potassium
    Ar = 40   #: Argon
    Ca = 40   #: Calcium
    Sc = 45   #: Scandium
    Ti = 48   #: Titanium
    V = 51    #: Vanadium
    Cr = 52   #: Chromium
    Mn = 55   #: Manganese
    Fe = 56   #: Iron
    Co = 59   #: Cobalt
    Ni = 58   #: Nickel
    Cu = 63   #: Copper
    Zn = 64   #: Zinc
    Ga = 69   #: Gallium
    Ge = 74   #: Germanium
    As = 75   #: Arsenic
    Se = 80   #: Selenium
    Br = 79   #: Bromine
    Kr = 84   #: Krypton
    Rb = 85   #: Rubidium
    Sr = 88   #: Strontium
    Y = 89    #: Yttrium
    Zr = 90   #: Zirconium
    Nb = 93   #: Niobium
    Mo = 98   #: Molybdenum
    Tc = 98   #: Technetium
    Ru = 102  #: Ruthenium
    Rh = 103  #: Rhodium
    Pd = 106  #: Palladium
    Ag = 107  #: Silver
    Cd = 114  #: Cadmium
    In = 115  #: Indium
    Sn = 120  #: Tin
    Sb = 121  #: Antimony
    Te = 130  #: Tellurium
    I = 127   #: Iodine   # noqa: E741
    Xe = 132  #: Xenon
    Cs = 133  #: Cesium
    Ba = 138  #: Barium
    La = 139  #: Lanthanum
    Ce = 140  #: Cerium
    Pr = 141  #: Praseodymium
    Nd = 142  #: Neodymium
    Pm = 145  #: Promethium
    Sm = 152  #: Samarium
    Eu = 153  #: Europium
    Gd = 158  #: Gadolinium
    Tb = 159  #: Terbium
    Dy = 164  #: Dysprosium
    Ho = 165  #: Holmium
    Er = 166  #: Erbium
    Tm = 169  #: Thulium
    Yb = 174  #: Ytterbium
    Lu = 175  #: Lutetium
    Hf = 180  #: Hafnium
    Ta = 181  #: Tantalum
    W = 184   #: Tungsten
    Re = 187  #: Rhenium
    Os = 192  #: Osmium
    Ir = 193  #: Iridium
    Pt = 195  #: Platinum
    Au = 197  #: Gold
    Hg = 202  #: Mercury
    Tl = 205  #: Thallium
    Pb = 208  #: Lead
    Bi = 209  #: Bismuth
    Th = 232  #: Thorium
    Pa = 231  #: Protactinium
    U = 238   #: Uranium


class SRSRGA(Device):
    # Misc
    clear = Cpt(EpicsSignal, ':Clear')
    init = Cpt(EpicsSignal, ':Init', kind="omitted")
    get_id = Cpt(EpicsSignalRO, ':getID', kind="omitted")

    # Ionizer control
    stop_filament = Cpt(EpicsSignal, ':StopFilament')
    set_degas_time = Cpt(EpicsSignal, ':SetDegasTime')
    get_degas_time = Cpt(EpicsSignalRO, ':GetDegasTime')
    set_electron_energy = Cpt(EpicsSignal, ':SetElectronEnergy')
    get_electron_energy = Cpt(EpicsSignalRO, ':GetElectronEnergy')
    set_emission_current = Cpt(EpicsSignal, ':SetEmissionCurrent')
    get_emission_current = Cpt(EpicsSignalRO, ':GetEmissionCurrent')

    is_filament_on = Cpt(EpicsSignal, ':IsFilamentOn')
    filament_timer = Cpt(EpicsSignal, ':FilamentTimer')

    set_ion_energy = Cpt(EpicsSignal, ':SetIonEnergy')
    get_ion_energy = Cpt(EpicsSignalRO, ':GetIonEnergy')
    ion_energy = Cpt(EpicsSignal, 'GetIonEnergy', write_pv=':SetIonEnergy', kind='normal')

    set_focus_plate_voltage = Cpt(EpicsSignal, ':SetFocusPlateVoltage')
    get_focus_plate_voltage = Cpt(EpicsSignalRO, ':GetFocusPlateVoltage')

    # Detection control
    set_noise_floor = Cpt(EpicsSignal, ':SetNoiseFloorLevel')
    get_noise_floor = Cpt(EpicsSignalRO, ':GetNoiseFloorLevel')

    # Scan setup
    set_initial_mass = Cpt(EpicsSignal, ':SetInitialMass')
    get_initial_mass = Cpt(EpicsSignalRO, ':GetInitialMass')
    set_final_mass = Cpt(EpicsSignal, ':SetFinalMass')
    get_final_mass = Cpt(EpicsSignalRO, ':GetFinalMass')
    set_steps_per_mass = Cpt(EpicsSignal, ':SetStepsPerMass')
    get_steps_per_mass = Cpt(EpicsSignalRO, ':GetStepsPerMass')

    pressure_sensitivity = Cpt(EpicsSignal, ':SetPressureSensitivity')
    get_pressure_sensitivity = Cpt(EpicsSignalRO, ':GetPressureSensitivity')

    calibration_factor = Cpt(EpicsSignal, ':SetCalibrationFactor')

    # Scans
    analog_scan = Cpt(EpicsSignal, ':AnalogScan')
    analog_scan_total = Cpt(EpicsSignal, ':AnalogScanTotal')

    start_histogram_scan = Cpt(EpicsSignal, ':StartHistogramScan')
    histogram_scan = Cpt(EpicsSignal, ':HistogramScan')
    histogram_scan_total = Cpt(EpicsSignal, ':HistogramScanTotal')
    Averaged_histogram_scan = Cpt(EpicsSignal, ':AverageHistogramScan')

    scans_number = Cpt(EpicsSignalRO, ':ScansNumber')
    scans_flag = Cpt(EpicsSignalRO, ':ScanFlag')

    single_mass_scan = Cpt(EpicsSignal, ':SingleMassScan')
    single_mass_pressure = Cpt(EpicsSignalRO, ':SingleMassPressure')

    # EM control
    turn_em_on = Cpt(EpicsSignal, ':TurnEMOn')
    turn_em_off = Cpt(EpicsSignal, ':TurnEMOff')
    get_em_status = Cpt(EpicsSignalRO, ':GetEMStatus')

    def single_mass_scan_n_times(self, mass: AtomicMass | int, n_scans: int) -> list[float]:
        """
        Perform a single-mass scan multiple times and return the pressure data points.

        This method triggers the single-mass scan at the specified mass value `n_scans` times.
        Each scan is executed synchronously, and the resulting pressure reading is appended
        to a list if valid.

        Parameters
        ----------
        mass : int
            The mass-to-charge (m/z) value to scan.
        n_scans : int
            The number of scan repetitions to perform. Must be greater than zero.

        Returns
        -------
        list of float
            List of pressure readings from each successful scan.

        Raises
        ------
        AssertionError
            If `n_scans` is not a positive integer.

        Notes
        -----
        - This method uses `.put(..., wait=True)` to ensure the scan completes before continuing.
        - Output is printed to the terminal for each pressure value collected.

        Examples
        --------
        >>> from pcdsdevices.rga import SRSRGA, AtomicMass
        >>> chamber_rga = SRSRGA("TMO:DREAM:RGA:01", name="chamber_rga")
        >>> chamber_rga.single_mass_scan_n_times(AtomicMass.Ar, 10)
        """

        assert n_scans > 0, "number scans must be integer bigger than zero"
        assert self.is_filament_on.get() == 1, "Filament is off, please turn it on before starting a scan"
        assert self.get_em_status.get() == 1, "Electron multiplier is off, please turn it on before starting a scan"

        if isinstance(mass, AtomicMass):
            mass = mass.value
        data_points = []
        valid_scan = 0
        while valid_scan < n_scans+1:
            self.single_mass_scan.put(mass, wait=True)
            partial_pressure = self.single_mass_pressure.get()
            if partial_pressure is not None and partial_pressure > 0.0:
                logger.info(f"valid scan point # {valid_scan+1}: {partial_pressure:.3e}")
                valid_scan += 1
                data_points.append(partial_pressure)
            else:
                logger.warning(f"invalid pressure value {partial_pressure}")
        data_points = data_points[1::]  # sometimes first sample shows overshoot

    def average_multiple_scans(self, n_scans: int):
        assert n_scans > 0, "number scans must be integer bigger than zero"
        assert self.is_filament_on.get() == 1, "Filament is off, please turn it on before starting a scan"
        assert self.get_em_status.get() == 1, "Electron multiplier is off, please turn it on before starting a scan"
        self.histogram_scan.put(np.zeros(300))
        self.Averaged_histogram_scan.put(np.zeros(300))
        scans_data = []
        for scan_numnber in range(1, n_scans+1):
            self.start_histogram_scan.put(1)
            _wait_for_value(self.scans_flag, 1, poll_time=1, timeout=720)
            _wait_for_value(self.scans_flag, 0, poll_time=1, timeout=720)
            scans_data.append(self.histogram_scan.get())
            self.Averaged_histogram_scan.put(np.mean(scans_data, axis=0))
            logger.info(f"scan #{scan_numnber} fininshed")
        
        return scans_data
    
    
dream_chamber_rga = SRSRGA("TMO:DREAM:RGA:01", name="dream_chamber_rga")
dream_catcher_rga = SRSRGA("TMO:DREAM:RGA:02", name="dream_catcher_rga")