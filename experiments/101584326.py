from pcdsdevices.pv_positioner import PVPositionerIsClose
from pcdsdevices.device import Device
from ophyd.signal import EpicsSignal, EpicsSignalRO, Signal
from pcdsdevices.interface import BaseInterface
from ophyd.device import Component as Cpt
from tmo.tmo_utils import log_positions as _log_positions

class MbesMotor(PVPositionerIsClose):
    setpoint = Cpt(EpicsSignal, '')
    readback = Cpt(EpicsSignalRO, '.RBV')
    atol = 0.01
    
    
class User(BaseInterface, Device):
    
    tab_component_names= True
    tab_whitelist = ['get_user']#, 'llt', 'lxt_stc']
    
    tmo_mbes_magnet_x = Cpt(MbesMotor, 'TMO:MBES:MMS:01')
    tmo_mbes_magnet_y = Cpt(MbesMotor, 'TMO:MBES:MMS:02')
    tmo_mbes_magnet_z = Cpt(MbesMotor, 'TMO:MBES:MMS:03')
    
    def __init__(self):
        name = 'user'
        return super().__init__(name=name)