from ophyd import EpicsSignalRO, EpicsSignal, Device
from ophyd import Component as Cpt
from ophyd.status import DeviceStatus
import time
import logging
logger = logging.getLogger(__name__)

class Bykiks(Device):
    """
    Control BYKIKS configuration.
    """
    abort_enable = Cpt(EpicsSignal, 'IOC:IN20:EV01:BYKIKS_ABTACT')
    abort_every_n = Cpt(EpicsSignal, 'IOC:IN20:EV01:BYKIKS_ABTPRD')
    burst_mode_enable = Cpt(EpicsSignalRO, 'IOC:BSY0:MP01:REQBYKIKSBRST')
    shots_to_burst = Cpt(EpicsSignal, 'PATT:SYS0:1:BYKIKSCNTMAX')
    request_burst =  Cpt(EpicsSignal, 'PATT:SYS0:1:BYKIKSCTRL')
    
    def trigger(self):
#        while bool(self.request_burst.get()): #wait for previous trigger burst to stop
#            time.sleep(0.01)
        time.sleep(1) #give additional time for the DAQ to configure when running with bluesky
        logger.info('Burst Enable')
        self.request_burst.put(1) #request burst
        return DeviceStatus(self, done=True, success=True)

#    def stage(self): #This is called at the beginning of a bluesky scan 
#        time.sleep(10) #this is needed to give the DAQ time to configure before bursting. 
#        retrun[self]

bykiks = Bykiks('', name='bykiks', read_attrs=['burst_mode_enable'])


#Define a function to set bykiks
def set_bykik(bykik):
    """
    Function to enable bykik and set to abort every N shots. Using 0 will stop bykik, -1 has not effect. 
    """
    if (bykik>0):
        bykiks.abort_every_n.put(bykik)
        bykiks.abort_enable.put(1)
    elif (bykik == 0): bykiks.abort_enable.put(0)

def get_bykik():
    """
    This function gets the status of bykik.
    """
    msg = "bykik is set to abort every " + str(bykiks.abort_every_n.get()) + " shots, and abort enable is set to " + str(bool(bykiks.abort_enable.get()))
    print (msg)
