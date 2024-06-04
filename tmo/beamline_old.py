from hutch_python.utils import safe_load
import numpy as np
import sys
from epics import caget, caput, cainfo
from tmo.db import im1k4, im2k4, im3k4, im4k4, im5k4
sys.path.append('/cds/home/opr/tmoopr')

import las_scan_TW as LAS


with safe_load('Vacuum Checkout'):
    from tmo.vac_checkout import tmo_vac_checkout

with safe_load('TMO Scans'):
    from tmo.tmo_scans import move_and_wait, imprint_scan, las_scan_JPC

with safe_load('FS14 lxt, txt, and lxt_ttc'):
    from tmo.lxt2 import lxt, txt, lxt_ttc

with safe_load('TMO Sequencers'):
    from tmo.tmo_sequencer import tmo_sequencer1, tmo_sequencer8, gas_jet_sequence, goose_trigger, burst_trigger
#    import tmo.tmo_sequencer

with safe_load('END BEAMTIME'):
    from tmo.end_beamTime import transmission_set, transmission_set_fraction, andor_on_off, end_of_beamtime, imagers_out, open_valves_all, close_valves_all

with safe_load('User Sequences'):
    import tmo.user_seq as user_seq

with safe_load('BYKIKS'):
    from tmo.bykiks import bykiks

with safe_load('CAM Recorder'):
    from tmo.cam_to_file import h5_img_collect, ppm_scan

with safe_load('TMO MPOD01'):
    from tmo.tmo_mpods import *

with safe_load('Waveplate Scan Motor'):
    from pcdsdevices.pv_positioner import PVPositionerIsClose
    from ophyd.device import Component as Cpt
    from ophyd.signal import EpicsSignal, EpicsSignalRO
    class WaveplateMotor(PVPositionerIsClose):
        setpoint = Cpt(EpicsSignal, ':MOVE')
        readback = Cpt(EpicsSignalRO, ':CURPOS')
        atol = 0.1
        rtol = 0
    QWP = WaveplateMotor('LM1K4:EJX_MP2_ATT1_POL1:ELL:M1', name='QWP')


with safe_load('LCLS-II get_daq - Timed Run trial version'):
    #def get_daq():
    #    from tmo.tmo_daq import daq
    #    return daq
    from tmo.tmo_daq_tmp import get_daq
    #daq = get_daq()

    def _timedRun(daq, duration=10, record=True):
        daq.set_configured_state()
        daq.control.setRecord(record)        
        daq.set_running_state()
        daq.sleep(duration)
        daq.set_configured_state()
        daq.control.setRecord(False)

    def timedRun(daq, duration=10, nrep=1, record=True):
        """
        Function starts a series of DAQ runs with a time limit.
        Arguments:
        duration: Duration of individual DAQ run [s]
        nrep    : Number of individual DAQ runs
        record: Boolean determining recording of the run.
        """
        try:
            for irep in range(nrep):
                _timedRun(daq, duration, record)
        except KeyboardInterrupt:
            daq.set_configured_state()

    def untimedRun(daq, record=True):
        """
        Function starts a DAQ run without a time limit.
        Arguments:
        daq   : DAQ object
        record: Boolean determining recording of the run.
        """
        daq.set_configured_state()
        daq.control.setRecord(record)        
        daq.set_running_state()
        print('daq.set_configured_state() to stop run')
    
    from tmo.db import bp, RE
    def step_scan(daq, motor, start, stop, nstep, nevents_step):
        #print('no explicit step counting yet - this may be necessary for the DAQ too properly wait for the events....')
        #print('need to test which state the DAQ can be infor this to work: expose the control level?')
        dets = [daq]
        daq.readoutCount = nevents_step
        daq.configure(motors=[motor])
        start_position = motor.wm()
        RE(bp.scan(dets, motor, start, stop, nstep))
        daq.control.setRecord(False)
        motor.mv(start_position)

    def list_scan(daq, motor, postList, nevents_step, record=True):
        #print('no explicit step counting yet - this may be necessary for the DAQ too properly wait for the events....')
        #print('need to test which state the DAQ can be infor this to work: expose the control level?')
        if isinstance(postList, np.ndarray):
            postList = postList.tolist()
        if not isinstance(postList, list):
            print('you need to lass a list of steps')
            return      
        daq.control.setState('connected')
        #daq.control.setRecord(record)
        daq.configure(record=record)        
        dets = [daq]
        daq.readoutCount = nevents_step
        daq.configure(motors=[motor])
        start_position = motor.wm()
        RE(bp.list_scan(dets, motor, postList))
        daq.control.setRecord(False)
        motor.mv(start_position)

    def random_scan(daq, motor, posList, nevents_step, tot_events, record=True):
        
        if isinstance(posList, np.ndarray):
            posList = posList.tolist()
        if not isinstance(posList, list):
            print('you need to lass a list of steps')
            return

        fullList = []
        num_data = 0
        while num_data < tot_events:
            np.random.shuffle(posList)
            fullList.extend(posList)    
            num_data += nevents_step * len(posList)
   
        list_scan(daq, motor, fullList, nevents_step, record) 
 

with safe_load('continous scan'):
    from tmo.continuous_scan import continuous_scan
