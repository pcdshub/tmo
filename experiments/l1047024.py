from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from ophyd.signal import EpicsSignal, EpicsSignalRO
from ophyd.device import Component as Cpt
from ophyd.device import FormattedComponent as FCpt
from ophyd.pv_positioner import PVPositioner
from pcdsdevices.interface import FltMvInterface
from tmo.tmo_utils import log_positions as _log_positions
from tmo.tmo_beamline_control import unds
import numpy as np


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

class SXRSS_Chicane(FltMvInterface, PVPositioner):
    
    tab_component_names= True
    tab_whitelist = ['chicane_delay', 'und_delay', 'estimate_delay']
    
    setpoint = FCpt(EpicsSignal, 'RIX:USER:MCC:EPHOTK:SET2', kind = 'hinted')
    readback = FCpt(EpicsSignalRO, 'SIOC:SYS0:ML07:AO653', kind = 'hinted')
    done = FCpt(EpicsSignalRO,'SIOC:SYS0:ML07:AO652', kind='normal',
                doc='PV that is 0 while the motors are moving and 1 when device is ready for a new request.')
    done_value = 1
    chicane_bend_bact = Cpt(EpicsSignalRO, 'BEND:UNDS:3510:BACT', kind = 'omitted')
        
    @property
    def chicane_delay(self):
        return self.readback.get()
    
    def und_delay(self, t0=0.140, del_und=None):
        delay = t0
        num_und=1
        kinit = unds.u36.k_act.get()
        u=37
        k=unds.u37.k_act.get()
        
        while np.isclose(kinit, k, rtol=0.02):
            num_und+=1
            u+=1
            uN = getattr(unds, 'u%s'%u)
            k = uN.k_act.get()
        
        if del_und is not None:
            delay += del_und * num_und
        else:
            delay += 0.125 * ((1+kinit**2/2)/4.7514) * num_und
        return delay
    
    @property
    def estimate_delay(self, **kwargs):        
        return self.chicane_delay + self.und_delay(**kwargs)
       
    @property
    def bact_delay(self, dispersion=2.221202E2):
        return dispersion * (self.chicane_bend_bact.get())**2
         
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
    sxrss_chicane = Cpt(SXRSS_Chicane, '')
    
    def __init__(self):
        name = 'user'
        return super().__init__(name=name)
        

 