from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from ophyd.signal import EpicsSignal, EpicsSignalRO
from pcdsdevices.epics_motor import SmarAct
from pcdsdevices.pseudopos import SyncAxis, delay_instance_factory
from ophyd.device import Component as Cpt
from ophyd.device import FormattedComponent as FCpt
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
    _voltage_set = Cpt(EpicsSignalRO, ':GetVoltagaeSetvalue')
    current = Cpt(EpicsSignal, ':GetCurrent', write_pv=':SetCurrent', kind='normal')
    curr_ramp = Cpt(EpicsSignal, ':GetCurrentRampSpeed', write_pv=':SetCurrentRampSpeed', kind='normal')
    _current_set = Cpt(EpicsSignalRO, ':GetCurrentSetvalue')
    status = Cpt(EpicsSignal, ':GetStatus')
    _on_switch = Cpt(EpicsSignal, ':SetSwitchONOFF', kind='omitted')
    
    def on(self):
        self._on_switch.put(1)
        
    def off(self):
        self._on_switch.put(0)

class SXRSS_Chicane(BaseInterface, Device):
    
    setpoint = FCpt(EpicsSignal, 'RIX:USER:MCC:EPHOTK:SET2', kind = 'hinted')
    readback = FCpt(EpicsSignalRO, 'SIOC:SYS0:ML07:AO653', kind = 'hinted')
    done = FCpt(EpicsSignalRO,'SIOC:SYS0:ML07:AO652', kind='normal',
                doc='PV that is 0 while the motors are moving and 1 when device is ready for a new request.')
    done_value = 1

    @property
    def chicane_delay(self):
        return self.readback.get()
    
    def und_delay(self):
        return 0
    
    def estimate_delay(self):
        return self.chicane_delay
        
         
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
    
    def __init__(self):
        name = 'user'
        return super().__init__(name=name)
        

            
    
    
    
    