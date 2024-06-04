import sys
sys.path.append('/cds/home/opr/tmoopr/hutch-python/tmo/')
from tmo.end_beamTime import transmission_set
from tmo.db import im2k4, tmo_lamp
from beamline import timedRun, untimedRun
from elog import HutchELog as elog_temp


def darkRun (daq, duration=10):
    elog = elog_temp.from_registry()
    im2k4.target('YAG1', wait=True) 
    timedRun(daq, duration, nrep=1, record=True)
    run_number = daq.control.getStatus()[-1]
    print('Dark Run Finished; Run = ',run_number)
    elog.post('epix100 Dark Run', run=run_number)
    untimedRun(daq, record=False)    
    im2k4.target('OUT', wait=True)
    
    

def specRun (daq, duration=10):
    # This function moves the epix100 detector out of the beam, detector_y to -1. Then records a run. After the run, it puts the epix100 back in place. 
    #
    elog = elog_temp.from_registry()
    im2k4.target('YAG1')
    transmission_set(10) #set transmission to 10%
    tmo_lamp.detector_y.mv(-1, wait=True)
    im2k4.target('OUT', wait=True)
    timedRun(daq, duration, nrep=1, record=True)
    untimedRun(daq, record=False)
    run_number = daq.control.getStatus()[-1]
    elog.post('Spectrometer Data Run', run=run_number)
    im2k4.target('YAG1')
    transmission_set(100) #set transmission to 100%
    tmo_lamp.detector_y.mv(-27.451, wait=True)
    im2k4.target('OUT')
       


