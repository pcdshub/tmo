from hutch_python.utils import safe_load
import numpy as np

from ophyd.device import Component as Cpt
from ophyd.device import FormattedComponent as FCpt
from ophyd.signal import EpicsSignal, EpicsSignalRO
from ophyd.pv_positioner import PVPositioner
from pcdsdevices.interface import FltMvInterface

class MagnetMotor1(FltMvInterface, PVPositioner):
        setpoint = FCpt(EpicsSignal, 'TMO:USER:MCC:EPHOTK:SET2', kind = 'hinted')
        readback = FCpt(EpicsSignalRO, 'SIOC:SYS0:ML04:AO905', kind = 'hinted')
        atol = 0.1
        rtol = 0
        done = FCpt(EpicsSignal,'SIOC:SYS0:ML07:AO651', kind='normal',
            doc=(
                'PV that is 0 while the motors are moving and 1 when device is '
                'ready for a new request.'
            )
        )
        done_value = 1

xcpepx_um = MagnetMotor1(name="xcpepx_um")

class DavidMotor(FltMvInterface, PVPositioner):
        setpoint = FCpt(EpicsSignal, 'RIX:USER:MCC:EPHOTK:SET2', kind = 'hinted')
        readback = FCpt(EpicsSignalRO, 'SIOC:SYS0:ML07:AO653', kind = 'hinted')
        atol = 0.1
        rtol = 0
        done = FCpt(EpicsSignal,'SIOC:SYS0:ML07:AO652', kind='normal',
            doc=(
                'PV that is 0 while the motors are moving and 1 when device is '
                'ready for a new request.'
            )
        )
        done_value = 1
#  statusPV='SIOC:SYS0:ML07:AO652'; 
#  remotePV='RIX:USER:MCC:EPHOTK:SET2';
#readbackPV='SIOC:SYS0:ML07:AO653';
#     setPV='SIOC:SYS0:ML07:AO654';

peppex_y = DavidMotor(name="peppex_y")

class User:
    def __init__(self):
        self.xcpepx_um = MagnetMotor1(name="xcpepx_um")
        self.peppex_y = DavidMotor(name="peppex_y")
