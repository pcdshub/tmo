from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from ophyd.signal import EpicsSignal, EpicsSignalRO
from pcdsdevices.epics_motor import SmarAct
from pcdsdevices.pseudopos import SyncAxis, delay_instance_factory
from ophyd.device import Component as Cpt
from pcdsdevices.device import ObjectComponent as OCpt
from tmo.tmo_utils import log_positions as _log_positions
from tmo.lxt2 import lxt

class BKP(BaseInterface, Device):
    
    tab_component_names= True
    tab_whitelist = []
    
    voltage = Cpt(EpicsSignal, ':AVDD_VOLT', write_pv=':SETVOLTAGE_RAW_IN', kind='normal')
    current = Cpt(EpicsSignal, ':AVDD_CURR', write_pv=':SETCURRENT_RAW_IN', kind='normal')
    volttage_limit = Cpt(EpicsSignal, ':LIMIT_V', write_pv=':SETUPLIMIT_IN', kind='omitted')
    mode = Cpt(EpicsSignalRO, ':MODE', kind='normal')
    max_voltage = Cpt(EpicsSignalRO,':MAX_V', kind='normal')
    max_current = Cpt(EpicsSignalRO, ':MAX_I', kind='normal')
    _set_on = Cpt(EpicsSignal, ':SETAVDDON', kind='omitted')
    _set_off = Cpt(EpicsSignal, ':SETAVDDOFF', kind='omitted')
    
    def on(self):
        self._set_on.put(1)
        
    def off(self):
        self._set_off.put(1)
 
class FPS(BaseInterface, Device):
    
    tab_component_names= True
    tab_whitelist = ['on', 'off']
    
    voltage = Cpt(EpicsSignal, ':GetVoltage',write_pv=':SetVoltage', kind='normal')
    volt_ramp = Cpt (EpicsSignal, ':GetVoltageRampSpeed', write_pv=':SetVoltageRampSpeed', kind='normal')
    voltage_set = Cpt(EpicsSignalRO, ':GetVoltagaeSetvalue', kind='omitted')
    current = Cpt(EpicsSignal, ':GetCurrent', write_pv=':SetCurrent', kind='normal')
    curr_ramp = Cpt(EpicsSignal, ':GetCurrentRampSpeed', write_pv=':SetCurrentRampSpeed', kind='normal')
    current_set = Cpt(EpicsSignalRO, ':GetCurrentSetvalue', kind='omitted')
    status = Cpt(EpicsSignal, ':GetStatus')
    _on_switch = Cpt(EpicsSignal, ':SetSwitchONOFF', kind='omitted')
    
    def on(self):
        self._on_switch.put(1)
        
    def off(self):
        self._on_switch.put(0)

class Chicane(BaseInterface, Device):
    
    tab_component_names= True
    
    bend_bact = Cpt(EpicsSignalRO, ':BACT')
    bend_bdes = Cpt(EpicsSignalRO, ':BDES')
    bend_bctrl = Cpt(EpicsSignalRO, ':BCTRL')    
      
class User(BaseInterface, Device):
    
    tab_component_names= True
    tab_whitelist = []
    
    bk01 = Cpt(BKP, 'TMO:BKP:01')
    bk02 = Cpt(BKP, 'TMO:BKP:02')
    bk03 = Cpt(BKP, 'TMO:BKP:03')
    bk04 = Cpt(BKP, 'TMO:BKP:04')
    fps01 = Cpt(FPS, 'TMO:FPS:01')
    fps02 = Cpt(FPS, 'TMO:FPS:02')
    fps03 = Cpt(FPS, 'TMO:FPS:03')
    sxrss_chicane = Cpt(Chicane, 'BEND:UNDS:3510')
    
    def __init__(self):
        name = 'user'
        return super().__init__(name=name)
    
    
    
    
    
        