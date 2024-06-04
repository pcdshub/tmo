"""
Standard classes for LCLS Energy Monitors.
"""
import logging

from ophyd import Component as Cpt
from ophyd import Device
from ophyd.signal import SignalRO, EpicsSignalRO

from pcdsdevices.interface import BaseInterface

logger = logging.getLogger(__name__)


class GEM(BaseInterface, Device):
    """
    Gas Energy Monitor from the LUSI project.

    A base class for a Gas Energy Monitor.

    Parameters
    ----------
    prefix : str
        Full GEM base PV.

    name : str
        Name to refer to the GEM.
    """

    not_implemented = Cpt(SignalRO, name="Not Implemented",
                          value="Not Implemented", kind='normal')


class GMD(BaseInterface, Device):
    """
    Gas Monitor Detector, instrument designed for precision measurements
    of beam intensity.

    Note: same class can be used for GMD and XGMD

    Parameters
    ----------
    prefix : str
        Full GMD base PV.

    name : str
        Name to refer to the GMD.
    """

    average_pulse_intensity = Cpt(EpicsSignalRO, ':HPS:AvgPulseIntensity',
                                  kind='hinted', doc='Avg Pulse energy [mJ]')
    energy_per_pulse = Cpt(EpicsSignalRO, ':HPS:milliJoulesPerPulse',
                           kind='hinted', doc='Pulse energy [mJ]')
    average_photons_per_pulse = Cpt(EpicsSignalRO, ':HPS:AvgPhotonsPerPulse',
                                    kind='hinted',
                                    doc='average number of photons per pulse')
    average_transmission = Cpt(EpicsSignalRO, ':HPS:AvgTransmission',
                               kind='hinted', doc='average transmission')
    gas_type = Cpt(EpicsSignalRO, ':GAS_TYPE_RBV', doc='Gas Type', string=True)
