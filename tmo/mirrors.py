#!/usr/bin/env python

from ophyd import Component as Cpt
from ophyd import Device
from ophyd.signal import EpicsSignal, EpicsSignalRO
from pcdsdevices.energy_monitor import GMD
from pcdsdevices.interface import BaseInterface
from pcdsdevices.mirror import KBOMirror
from pcdsdevices.pim import PPM, LCLS2ImagerBase
import dataclasses

from tmo.db import im2k0  # FEE imager
from tmo.db import im3k4, im4k4, im5k4, im6k4  # TMO imagers
from tmo.db import mr2k4_kbo, mr3k4_kbo, mr4k4_kbo, mr5k4_kbo  # TMO mirrors


class XGMD(GMD):
    """
    X-ray Gas Monitor Detector (XGMD) class, inheriting properties from
    GMD class.
    """
    pass


@dataclasses.dataclass
class Mirror:
    """
    Class representing mirror configurations and related objects.

    Parameters:
    -----------
    name : str
        Name of the mirror configuration.
    beamline : str
        Name of the beamline associated with the mirror.
    imager_before : PPM
        PPM (Power/Position Monitor) before the mirror.
    imager_after : PPM
        PPM after the mirror.
    attenuator : EpicsSignalRO
        Attenuator signal for the mirror.
    gmd : GMD
        GMD associated with the mirror.
    xgmd : XGMD
        XGMD associated with the mirror.
    transmission : EpicsSignalRO
        Transmission signal for the mirror.
    mirror_objs : list
        List of KBOMirror objects associated with the mirror.
    """
    name: str
    beamline: str
    imager_before: PPM
    imager_after: PPM
    attenuator: EpicsSignalRO
    gmd: GMD
    xgmd: XGMD
    transmission: EpicsSignalRO
    mirror_objs: [KBOMirror]
    ppm_scale_factor: float


class MiscSignals(BaseInterface, Device):
    kfe_attenuator = Cpt(EpicsSignalRO, "AT1K0:GAS:TRANS_RBV")
    kfe_gmd = Cpt(GMD, "EM1K0:GMD")
    kfe_xgmd = Cpt(XGMD, "EM2K0:XGMD")
    kfe_rate = Cpt(EpicsSignalRO, "TPG:SYS0:1:DST04:RATE")
    kfe_photon_energy = Cpt(EpicsSignalRO, "SIOC:SYS0:ML00:AO628")
    kfe_im2k0 = Cpt(LCLS2ImagerBase, "IM2K0:XTES")


kfe_signals = MiscSignals(name="kfe_signals")


mirrors = {}
mirrors["TMO_KBO1"] = Mirror(name="TMO_KBO1",
                             beamline="TMO",
                             imager_before=im3k4,
                             imager_after=im4k4,
                             attenuator=kfe_signals.kfe_attenuator,
                             gmd=kfe_signals.kfe_gmd,
                             xgmd=kfe_signals.kfe_xgmd,
                             transmission=EpicsSignal("TMO:KBO1:Transmisstion"),
                             mirror_objs=[mr2k4_kbo, mr3k4_kbo],
                             ppm_scale_factor=0.0751)

mirrors["TMO_KBO2"] = Mirror(name="TMO_KBO2",
                             beamline="TMO",
                             imager_before=im5k4,
                             imager_after=im6k4,
                             attenuator=kfe_signals.kfe_attenuator,
                             gmd=kfe_signals.kfe_gmd,
                             xgmd=kfe_signals.kfe_xgmd,
                             transmission=EpicsSignal("TMO:KBO2:Transmisstion"),
                             mirror_objs=[mr4k4_kbo, mr5k4_kbo],
                             ppm_scale_factor=0.0721)
