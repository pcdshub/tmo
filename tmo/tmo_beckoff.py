from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from ophyd.device import Component as Cpt
from ophyd.device import FormattedComponent as FCpt
from ophyd.signal import EpicsSignal, EpicsSignalRO
from collections import OrderedDict
from tmo.tmo_utils import log_positions as _log_positions
from tmo.tmo_utils import add_tags as _add_tags
from tmo.tmo_utils import EpicsSignalAlarm

class BeckoffAo(BaseInterface, Device):
    tab_component_names = True
    tab_whitelist = ['readback']
    
    setpoint = FCpt(EpicsSignal, '{prefix}:AOUT:{ch}', kind='normal')
    desc = FCpt(EpicsSignal, '{prefix}:AOUT:{ch}.DESC', string=True, kind='normal')
    
    def __init__(self, prefix, ch=None, **kwargs):
        self.ch = ch
        return super().__init__(prefix, **kwargs)  
    
    @property
    def value(self):
        return self.setpoint.get()

class BeckoffDo(BaseInterface, Device):
    tab_component_names = True
    tab_whitelist = ['on', 'off', 'value']
    
    output = FCpt(EpicsSignal, '{prefix}:BOUT:{ch}', kind='normal')
    desc = FCpt(EpicsSignal, '{prefix}:BOUT:{ch}.DESC', string=True, kind='normal')
    dur = FCpt(EpicsSignal, '{prefix}:BOUT:{ch}.HIGH', kind='normal')

    def __init__(self, prefix, ch=None, **kwargs):
        self.ch = ch
        return super().__init__(prefix, **kwargs)  
    
    def on(self):
        self.output.put(1)
    
    def off(self):
        self.output.put(0)
    
    @property
    def value(self):
        return self.output.get()
        
class BeckhoffTcSingle(BaseInterface, Device):
    tab_component_names= True
    tab_whitelist = ['set_alarm', 'disp_alarm', 'value']
    
    temp = FCpt(EpicsSignalRO, '{prefix}:TC:{ch}', kind='normal')
    desc = FCpt(EpicsSignal, '{prefix}:TC:{ch}.DESC', string=True, kind='normal') 
    alarm = FCpt(EpicsSignalAlarm, '{prefix}:TC:{ch}', kind='normal' )
    
    def __init__(self, prefix, ch=None, **kwargs):
        self.ch = ch
        return super().__init__(prefix, **kwargs)  
           
    @property
    def disp_alarm(self):
        self.alarm.get()
    
    @property
    def value(self):
        return self.temp.get()
  
    
class BeckoffDm(BaseInterface, Device):
    tab_component_names = True
    tab_whitelist = ['value']
    
    readback = FCpt(EpicsSignalRO, '{prefix}:DMM:{ch}', kind='normal')
    desc =FCpt(EpicsSignal, '{prefix}:DMM:{ch}.DESC', string=True, kind='normal')  
    
    @property
    def value(self):
        return self.readback.get()
    

class Beckoff(BaseInterface, Device):
    tab_component_names = True
    tab_whitelist = ['summary']   
    
    aout_1 = Cpt(BeckoffAo, '', ch=1, kind='normal')
    aout_2 = Cpt(BeckoffAo, '', ch=2, kind='normal')
    aout_3 = Cpt(BeckoffAo, '', ch=3, kind='normal')
    aout_4 = Cpt(BeckoffAo, '', ch=4, kind='normal')
    aout_5 = Cpt(BeckoffAo, '', ch=5, kind='normal')
    aout_6 = Cpt(BeckoffAo, '', ch=6, kind='normal')
    aout_7 = Cpt(BeckoffAo, '', ch=7, kind='normal')
    aout_8 = Cpt(BeckoffAo, '', ch=8, kind='normal')
    aout_9 = Cpt(BeckoffAo, '', ch=9, kind='normal')
    
    dout_1 = Cpt(BeckoffDo, '', ch=1, kind='normal')
    dout_2 = Cpt(BeckoffDo, '', ch=2, kind='normal')
    dout_3 = Cpt(BeckoffDo, '', ch=3, kind='normal')
    dout_4 = Cpt(BeckoffDo, '', ch=4, kind='normal')
    dout_5 = Cpt(BeckoffDo, '', ch=5, kind='normal')
    dout_6 = Cpt(BeckoffDo, '', ch=6, kind='normal')
    dout_7 = Cpt(BeckoffDo, '', ch=7, kind='normal')
    dout_8 = Cpt(BeckoffDo, '', ch=8, kind='normal')
    
    tc_1 = Cpt(BeckhoffTcSingle, '', ch=1, kind='normal')
    tc_2 = Cpt(BeckhoffTcSingle, '', ch=2, kind='normal')
    tc_3 = Cpt(BeckhoffTcSingle, '', ch=3, kind='normal')
    tc_4 = Cpt(BeckhoffTcSingle, '', ch=4, kind='normal')
    tc_5 = Cpt(BeckhoffTcSingle, '', ch=5, kind='normal')
    tc_6 = Cpt(BeckhoffTcSingle, '', ch=6, kind='normal')
    tc_7 = Cpt(BeckhoffTcSingle, '', ch=7, kind='normal')
    tc_8 = Cpt(BeckhoffTcSingle, '', ch=8, kind='normal')
    
    def summary(self, log=False, msg=None, **kwargs):
        curr_settings=OrderedDict()
        for device_name in dir(self):
            device = getattr(self, device_name)
            if hasattr(device, 'desc'):
                desc = device.desc.get()
                if desc != '':
                    key = '%s (%s)' % (desc, device_name)
                    curr_settings[key] = device.value
        
        kwargs['tags'] = _add_tags(kwargs.get('tags', ''),'beckoff')      
        _log_positions(curr_settings, log=log, msg=msg, **kwargs)
        return                 

tmo_beckoff = Beckoff('TMO:USR:BHC', name='tmo_beckoff')    

def get_beckoff(log=False, msg=None, include_cam=None, **kwargs):
    """
    Get the User beckoff values
    """
    tmo_beckoff.summary(log, msg, include_cam=include_cam, **kwargs)
get_beckoff.__doc__ += _log_positions.__doc__ 