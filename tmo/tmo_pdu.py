from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from ophyd.device import Component as Cpt
from ophyd.device import FormattedComponent as FCpt
from ophyd.signal import EpicsSignal, EpicsSignalRO
from collections import OrderedDict
from tmo.tmo_utils import log_positions as _log_positions
from tmo.tmo_utils import add_tags as _add_tags


class PduSingleChannel(BaseInterface, Device):
    tab_component_names = True
    tab_whitelist = ['on', 'off', 'cycle']

    state = Cpt(EpicsSignal, ':getPduOutletState', write_pv=':setPduOutletCommand', kind='normal') 
    state_setpoint = Cpt(EpicsSignal, ':setPduOutletCommand', kind='omitted')
    ctrl_state = Cpt(EpicsSignalRO, ':getPduOutletControllable', kind='normal')    
    desc = Cpt(EpicsSignal, ':setPduOutletName', string=True, kind='normal')
    
    def on(self):
        self.state.put(2)
    
    def off(self):
        self.state.put(1)
        
    def cycle(self):
        self.state.put(3)
    
class PduEightPort(BaseInterface, Device):
    tab_component_names = True
    tab_whitelist = ['summary']
    
    ch1 = Cpt(PduSingleChannel, ':Outlet:1')
    ch2 = Cpt(PduSingleChannel, ':Outlet:2')
    ch3 = Cpt(PduSingleChannel, ':Outlet:3')
    ch4 = Cpt(PduSingleChannel, ':Outlet:4')
    ch5 = Cpt(PduSingleChannel, ':Outlet:5')
    ch6 = Cpt(PduSingleChannel, ':Outlet:6')
    ch7 = Cpt(PduSingleChannel, ':Outlet:7')
    ch8 = Cpt(PduSingleChannel, ':Outlet:8')
    
    def summary(self, log=False, msg=None, **kwargs):
        curr_settings=OrderedDict()
        status_enum = ['unknown', 'off', 'on']
        for ch in range(8):
            device = getattr(self, 'ch%s'%(ch+1))
            desc = device.desc.get()
            state = device.state.get()
            key = '%s(%s)' % (desc, device.name)
            curr_settings[key] = status_enum[state]
        _log_positions(curr_settings, log=False)
    
class TmoPdu(BaseInterface, Device):
    tab_component_names = True

    lamp_pdu = Cpt(PduEightPort, 'TMO:PWR:01', kind='normal')
    cvmi_pdu = Cpt(PduEightPort, 'TMO:PWR:CVMI', kind='normal')
    dream_pdu = Cpt(PduEightPort, 'TMO:PWR:DREAM', kind='normal')   
    mrco_pdu_09 = Cpt(PduEightPort, 'TMO:PWR:PR09', kind='normal')
    mrco_pdu_10 = Cpt(PduEightPort, 'TMO:PWR:PR10', kind='normal')
    
tmo_pdu = TmoPdu(name='tmo_pdu')