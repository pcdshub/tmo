from pcdsdevices.valve import VRC, VVC
from pcdsdevices.pump import PTMPLC
from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from ophyd.device import Component as Cpt
from ophyd.signal import EpicsSignal
from tmo.db import elog
import time
import sys
import logging
import datetime
logger = logging.getLogger(__name__)


#Pump States
# [ 0] STOPPED
# [ 1] STARTING
# [ 2] RUNNING
# [ 3] FAULT
# [ 4] STOPPING

class At1k0Reset(BaseInterface, Device):
    tab_component_names= True
    tab_whitelist = ['restart_pump', 'start_monitor', 'stop_monitor']
    
    ptm_120 = Cpt(PTMPLC,':PTM:120')
    ptm_130 = Cpt(PTMPLC,':PTM:130')
    vrc_120 = Cpt(VRC,':VRC:120')
    vrc_130 = Cpt(VRC,':VRC:130')
    vvc_ar = Cpt(VVC, ':VVC:71')
    
    def restart_pump(self, pump, valve):
        logger.info('Start %s' % pump.name)
        pump.switch_pump_on.put(1)
       
        while not pump.pump_at_speed.get():
            if pump.pump_state.get() != 1:
                pump.switch_pump_on.put(1)
                logger.info('%s did not restart, trying again' % pump.name)
            time.sleep(1)
        logger.info('%s started' % pump.name)
        
        logger.info('Open %s' % valve.name)
        time.sleep(1)
        valve.open_command.put(1)
        while valve.valve_position.get():
            time.sleep(1)
            valve.open_command.put(1)
    
        logger.info('Open %s' % self.vvc_ar.name)
        self.vvc_ar.open_command.put(1)
        
        while pump.pump_state.get() != 2:
            time.sleep(1) 
            
    
    def _monitor_cb_120(self, *args, value, timestamp, **kwargs):
        if not (value in [1,2]):
            logger.error('Pump 120 stopped!')
            #self.ptm_120.pump_state.unsubscribe_all()
            self.restart_pump(pump=self.ptm_120, valve=self.vrc_120)       
            #self.ptm_120.pump_state.subscribe(self._monitor_cb_120)
            #logger.warning('Pump 120 restarted, you need to run  user.at1k0_reset.start_monitor()')
            elog.post('Pump 120 stopped and was restarted.', tags='issues script-generated')
            return
        
    def _monitor_cb_130(self, *args, value, timestamp, **kwargs):
        if not (value in [1,2]): 
            logger.error('Pump 130 stopped!')
            #self.ptm_130.pump_state.unsubscribe_all()
            self.restart_pump(pump=self.ptm_130, valve=self.vrc_130)       
            #self.ptm_130.pump_state.subscribe(self._monitor_cb_130)
            #logger.warning('Pump 130 restarted, you need to run  user.at1k0_reset.start_monitor()')
            elog.post('Pump 130 stopped and was restarted.', tags='issues script-generated')
            
    def start_monitor(self):
        self.ptm_120.pump_state.subscribe(self._monitor_cb_120)
        self.ptm_130.pump_state.subscribe(self._monitor_cb_130)
        sys.stdout.write("\x1b]2;AT1K0 mon\x07")
        
    def stop_monitor(self):
        self.ptm_120.unsubscribe_all()
        self.ptm_130.unsubscribe_all()
    

class User(BaseInterface, Device):
    
    tab_component_names= True
    tab_whitelist = []
    
    at1k0_reset = Cpt(At1k0Reset, 'AT1K0:GAS')
    
    def __init__(self):
        name = 'user'
        return super().__init__(name=name)



    