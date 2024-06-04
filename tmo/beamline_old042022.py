from hutch_python.utils import safe_load
import numpy as np
import sys
import time as time
from epics import caget, caput, cainfo
from tmo.db import im1k4, im2k4, im3k4, im4k4, im5k4
from elog import HutchELog as elog_temp
sys.path.append('/cds/home/opr/tmoopr')

import las_scan_TW as LAS

with safe_load('Beam Energy Request'):
    from pcdsdevices.beam_stats import BeamEnergyRequest
    photon_energy1 = BeamEnergyRequest('TMO',name='photon_energy1',atol=0,skip_small_moves=False,bunch=1)
    photon_energy2 = BeamEnergyRequest('TMO',name='photon_energy2',atol=0,skip_small_moves=False,bunch=2)

with safe_load('Vacuum Checkout'):
    from tmo.vac_checkout import tmo_vac_checkout

with safe_load('TMO Scans'):
    from tmo.tmo_scans import move_and_wait, imprint_scan, las_scan_JPC

with safe_load('FS14 lxt, txt, and lxt_ttc'):
    from tmo.lxt2 import lxt, txt, lxt_ttc
    def shift_t0(shift):
        """
        this function zeros lxt_ttc and shifts lxt by the amount passed (in seconds). 
        """
        lxt_ttc.mv(0)
        lxt.mvr(shift)
        lxt_ttc.set_current_position(0)
        txt_position = txt.get()[2][0]
        msg = "moved lxt offset by " + str(shift*1E12) + "ps to compensate for drift of ATM. Current position of txt for lxt_ttc=0: " + str(txt_position) 
        elog = elog_temp.from_registry()
        elog.post(msg)
        return txt_position

with safe_load('TMO Sequencers'):
    from tmo.tmo_sequencer import tmo_sequencer1, tmo_sequencer8, gas_jet_sequence, goose_trigger, burst_trigger
#    import tmo.tmo_sequencer

with safe_load('END BEAMTIME'):
    from tmo.end_beamTime import transmission_set, transmission_set_fraction, andor_on_off, end_of_beamtime, imagers_out, open_valves_all, close_valves_all

with safe_load('User Sequences'):
    import tmo.user_seq as user_seq

with safe_load('BYKIKS'):
    from tmo.bykiks import bykiks
    def set_bykik(bykik):
        """
        Function to enable bykik and set to abort every N shots. Using -1 will stop bykik, 0 has not effect. 
        """
        if (bykik>0):
            bykiks.abort_enable.put(1)
            bykiks.abort_every_n.put(bykik)
        elif (bykik<0): bykiks.abort_enable.put(0)

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
    #from tmo.tmo_daq_scans import timedRun, untimedRun, stopRun, scan, timedRun2
    #JPC should uncomment the line above and remove all the functions below. 
    #def get_daq():
    #    from tmo.tmo_daq import daq
    #    return daq
    from tmo.tmo_daq_tmp import get_daq
    #daq = get_daq()

    def _timedRun(daq, duration=10, record=True, bykik=0):
        daq.control.setState('configured')
        prev_rec_state = daq.control.getStatus()[3] #attempt to make changing record state more stable        
        daq.control.setRecord(record)
        #daq.configure(record=record)
        set_bykik(bykik)
        daq.control.setState('running')
        print('record = ', daq.control.getStatus()[3])
        time.sleep(duration)
        daq.control.setState('configured')
        daq.control.setRecord(False)
        
    def timedRun(daq, duration=10, nrep=1, record=True, bykik=0, freeRun=False):
        """
        Function starts a series of DAQ runs with a time limit.
        Arguments:
        duration: Duration of individual DAQ run [s]
        nrep    : Number of individual DAQ runs
        record: Boolean determining recording of the run.
        bykik: set bykik to abort every N shots. bykik = -1 will stop bykik. bykik = 0 will not change the current status. 
        freeRun: boolean to start untimedRun after the timedRun is complete
        """
        try:
            for irep in range(nrep):
                _timedRun(daq, duration, record, bykik)
            print('Timed run has finished')
            print('\a')
            if freeRun:
                untimedRun(daq, False, bykik)
        except KeyboardInterrupt:
            daq.control.setState('configured')
          
    def untimedRun(daq, record=True, bykik=0):
        """
        Function starts a DAQ run without a time limit.
        Arguments:
        daq   : DAQ object
        record: Boolean determining recording of the run.
        bykik: set bykik to abort every N shots. bykik = -1 will stop bykik
        """
        daq.control.setState('configured')
        prev_rec_state = daq.control.getStatus()[3] #attempt to make changing record state more stable
        daq.control.setRecord(record)
        set_bykik(bykik)        
        daq.control.setState('running')
        print('daq.control.setState("configured") to stop run')
    
    def stopRun(daq, record=False):
        """
        This function stops the current run and sets the record state. 
        daq: DAQ object
        record: Boolean for the record state. 
        """
        daq.control.setState('configured')
        prev_rec_state = daq.control.getStatus()[3] #attempt to make changing record state more stable
        daq.control.setRecord(record)

    from tmo.db import bp, RE
    
    def timedRun2(daq, tot_events, nevents_cycle=3600, num_runs=1, record=True, bykik=0, freeRun=False):		
        """
        Function starts a series of DAQ runs with a time limit.
        Arguments:
        nevents_cycle: Number of events between screen updates, default is 30 sec at 120 Hz (3600 events). 
        tot_events: Total number of events in a run. 
        num_runs: Number of runs to take, each run will have tot_events
        record: Boolean determining recording of the run.
        bykik: set bykik to abort every N shots. bykik = -1 will stop bykik. bykik = 0 will not change the current status. 
        freeRun: boolean to start untimedRun after the timedRun is complete
        """           
        daq.control.setState('connected')
        daq.configure(record=record)        
        daq.configure(events=nevents_cycle)
        dets = [daq]
        set_bykik(bykik)
        
        if ( nevents_cycle > tot_events):
            print("Error: Requested more events/cycle than total events")
            return
        num_cycles = np.ceil(tot_events/nevents_cycle).astype(int);
        #The run engine should not go inside the for loop. Instead the loop should be executed as below then the RE called once
        def temp_plan(num_runs, num_cycles):
            for irep in range(num_runs):
                yield from bp.count(dets, num=num_cycles)     
        RE(temp_plan(num_runs, num_cycles))
        print('\a')
        if freeRun:
            untimedRun(daq, False, bykik)    

    class scan:
    
        def step(daq, motor, start, stop, nstep, nevents_step, record=True, bykik=0, restore=True):
            #print('no explicit step counting yet - this may be necessary for the DAQ too properly wait for the events....')
            #print('need to test which state the DAQ can be infor this to work: expose the control level?')
       
            #Setup DAQ and scan parameters
            dets = [daq]
            #daq.readoutCount = nevents_step
            daq.configure(events=nevents_step)
            daq.configure(record=record)
            daq.configure(motors=[motor])
            start_position = motor.position
        
            #BYKIK
            set_bykik(bykik)      

            #Scan
            RE(bp.scan(dets, motor, start, stop, nstep))
            daq.control.setRecord(False)
                    
            if (record): #Post scan details to elog. 
                msg = "Scan " + motor.name + " from " + str(start) + " to " + str(stop) + " in " + str(nstep) + " steps."    
                elog = elog_temp.from_registry()
                elog.post(msg,run = daq.control.getStatus()[-1])
            print('\a')
            if (restore):
                motor.mv(start_position) #Return to start position

        @staticmethod
        def list(daq, motor, posList, nevents_step, record=True, bykik=0, restore=True):
            #print('no explicit step counting yet - this may be necessary for the DAQ too properly wait for the events....')
            #print('need to test which state the DAQ can be infor this to work: expose the control level?')
        
            #Setup DAQ and scan parameters
            if isinstance(posList, np.ndarray):
                posList = posList.tolist()
            if not isinstance(posList, list):
                print('you need to lass a list of steps')
                return      
            daq.control.setState('connected')
            #daq.configure(motors=[motor], record=record, events=nevents_step)        
            dets = [daq]
            #daq.readoutCount = nevents_step
            daq.configure(events=nevents_step)
            daq.configure(motors=[motor])
            daq.configure(record=record)
            start_position = motor.position

            #BYKIK
            set_bykik(bykik)      

            #Scan
            RE(bp.list_scan(dets, motor, posList))
            daq.control.setRecord(False)
            #Post scan details to elog.
            if (record):  
                msg = "Scan " + motor.name + ": " + str( np.sort( np.unique( posList ) ) )
                elog = elog_temp.from_registry()
                elog.post(msg,run = daq.control.getStatus()[-1])
            print('\a')
            if (restore):
                motor.mv(start_position)

        def random(daq, motor, posList, nevents_step, tot_events, record=True, bykik=0, restore=True):
            """
            Random scan of a motor. This function will randomly scan the "motor" through the "posList" taking nevents_step per step until it has completed at least tot_events.
            bykik: set bykik to abort every N shots. bykik = -1 will stop bykik, bykik = 0 will not change the current setting. 
            restore: returns the motor to the starting position after the scan
            """
            if isinstance(posList, np.ndarray):
                posList = posList.tolist()
            if not isinstance(posList, list):
                print('you need to lass a list of steps')
                return

            #build the list of points to scan
            fullList = []
            num_data = 0
            while num_data < tot_events:
                np.random.shuffle(posList)
                fullList.extend(posList)    
                num_data += nevents_step * len(posList)

            #use list_scan to scan this list. 
            scan.list(daq, motor, fullList, nevents_step, record, bykik, restore) 

        def zigzag(daq, motor, posList, nevents_step, tot_events, record=True, bykik=0, restore=True):
            """
            zig-zag (or snake) scan of a motor. This function will scan the "motor" through the "posList" in a zig-zag fashion taking nevents_step per step until it has completed at least tot_events.
            bykik: set bykik to abort every N shots. bykik = -1 will stop bykik, bykik = 0 will not change the current setting. 
            restore: returns the motor to the starting position after the scan
            """
            if isinstance(posList, np.ndarray):
                posList = posList.tolist()
            if not isinstance(posList, list):
                print('you need to lass a list of steps')
                return

            #build the list of points to scan
            fullList = []
            num_data = 0
            posList.sort() #order the list of points sequentially
            while num_data < tot_events:
                #posList.reverse()
                fullList.extend(posList)    
                num_data += nevents_step * len(posList)
                posList.reverse()

            #use list_scan to scan this list. 
            scan.list(daq, motor, fullList, nevents_step, record, bykik, restore) 

with safe_load('continous scan'):
    from tmo.continuous_scan import continuous_scan
