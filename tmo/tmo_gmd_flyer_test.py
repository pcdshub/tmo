from pcdsdevices.interface import BaseInterface
from ophyd.signal import EpicsSignal, EpicsSignalRO
from ophyd.device import Component as Cpt
from pcdsdevices.device import UnrelatedComponent as UCpt
from pcdsdevices.device import GroupDevice, Device
from collections import OrderedDict
from ophyd.status import DeviceStatus
import time
import logging
logger = logging.getLogger(__name__)

class GMD(BaseInterface, Device):
    tab_component_names = True
    avg_int = Cpt(EpicsSignalRO, ':HPS:AvgPulseIntensity', kind='hinted',
             doc='Avg Pulse energy [mJ]')
    mj = Cpt(EpicsSignalRO, ':HPS:milliJoulesPerPulse', kind='hinted',
             doc='Pulse energy [mJ]')
    
#    avg_int = Cpt(EpicsSignalRO, 'SIOC:SYS0:ML00:AO628', kind='hinted',
#             doc='Avg Pulse energy [mJ]')
#    mj = Cpt(EpicsSignalRO, 'SIOC:SYS0:ML00:AO628', kind='hinted',
#             doc='Pulse energy [mJ]')
    
    def __init__(self, *args, **kwargs):
        self._completion_status = None #For fly scans
        self.flyer_attr=['mj','avg_int']
        self._start_time=None
        self._acquiring = False
        self._paused = False
        return super().__init__(*args, **kwargs)
    
    def _data_cb(self, value, timestamp, **kwargs):
        obj = kwargs['obj']
        self.data[obj.name]['value'].append(value)
        self.data[obj.name]['timestamp'].append(timestamp)
            
    def kickoff(self):
        logger.debug('called .kickoff')
        if self._completion_status is not None and not self._completion_status.done:
            raise RuntimeError("Kicking off a second time?!")      
        
        self.data={}#OrderedDict()
        for attr in self.flyer_attr:
            obj = getattr(self, attr)
            self.data[obj.name]={'value':[], 'timestamp':[]}
        
        st = DeviceStatus(device=self)
        self._completion_status = st
        
        for attr in self.flyer_attr:
            obj = getattr(self, attr)
            obj.subscribe(self._data_cb)
            logger.debug('subscribe %s' % obj.name)
        
        self._start_time = time.time()
        kickoff_st = DeviceStatus(device=self)
        kickoff_st.set_finished()
        return kickoff_st

    def pause(self):
        logger.debug('called .pause')
        
    def resume(self):
        logger.debug('called .resume')
        
    def complete(self):
        logger.debug('called .complete')
        if self._completion_status is None:
            raise RuntimeError("No collection in progress")
        
        for attr in self.flyer_attr:
            obj = getattr(self, attr)
            obj.unsubscribe_all()
            logger.debug('unsubscribe %s' % obj.name)
        
        self._completion_status.set_finished()
        return self._completion_status
    
    def collect(self):
        logger.debug('called .collect')
       
        timestamps={}
        value={}
        for attr, data in self.data.items():
            timestamps[attr]=data['timestamp']
            value[attr]=data['value']
        yield dict(time=self._start_time,
            timestamps=timestamps,
            data=value)
        
    def describe_collect(self):
        logger.debug('called .describe_collect')
        data={}
        for attr in self.flyer_attr:
            obj = getattr(self, attr)
            data[obj.name]={'source':obj.pvname, 'dtype':'number','shape':[], 'units':'mj'}
        return {self.name: data}
    
gmd_flyer = GMD('EM1K0:GMD', name='gmd')
#gmd_flyer = GMD('', name='gmd')