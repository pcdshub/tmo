from hutch_python.utils import safe_load
import numpy as np
import time as time
from elog import HutchELog as elog_temp
elog = elog_temp.from_registry()

from tmo.bykiks import bykiks
def _set_bykik(bykik):
    if (bykik>0):
        bykiks.abort_enable.put(1)
        bykiks.abort_every_n.put(bykik)
    elif (bykik == 0): bykiks.abort_enable.put(0)



from tmo.tmo_daq_tmp import get_daq

def _timedRun(daq, duration=10, record=True, bykik=-1):
    daq.control.setState('configured')
    prev_rec_state = daq.control.getStatus()[3] #attempt to make changing record state more stable        
    daq.control.setRecord(record)
    #daq.configure(record=record)
    _set_bykik(bykik)
    daq.control.setState('running')
    print('record = ', daq.control.getStatus()[3])
    time.sleep(duration)
    daq.control.setState('configured')
    daq.control.setRecord(False)
        
def timedRun(daq, duration=10, nrep=1, record=True, bykik=-1, freeRun=False):
    """
    Function starts a series of DAQ runs with a time limit.
    Arguments:
    duration: Duration of individual DAQ run [s]
    nrep    : Number of individual DAQ runs
    record: Boolean determining recording of the run.
    bykik: set bykik to abort every N shots. bykik = 0 will stop bykik. bykik = -1 will not change the current status. 
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
          
def untimedRun(daq, record=True, bykik=-1):
    """
    Function starts a DAQ run without a time limit.
    Arguments:
    daq   : DAQ object
    record: Boolean determining recording of the run.
    bykik: set bykik to abort every N shots. bykik = 0 will stop bykik. bykik = -1 will not change the current status.
    """
    daq.control.setState('configured')
    prev_rec_state = daq.control.getStatus()[3] #attempt to make changing record state more stable
    daq.control.setRecord(record)
    _set_bykik(bykik)        
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
    
def timedRun2(daq, tot_events, nevents_cycle=3600, num_runs=1, record=True, bykik=-1, freeRun=False, motor=None):		
    """
    Function starts a series of DAQ runs with a time limit.
    Arguments:
    nevents_cycle: Number of events between screen updates, default is 30 sec at 120 Hz (3600 events). 
    tot_events: Total number of events in a run. 
    num_runs: Number of runs to take, each run will have tot_events
    record: Boolean determining recording of the run.
    bykik: set bykik to abort every N shots. bykik = 0 will stop bykik. bykik = -1 will not change the current status. 
    freeRun: boolean to start untimedRun after the timedRun is complete
    motor: optional input of a motor name to append to the data
    """           
    daq.control.setState('connected')
    daq.configure(record=record)        
    daq.configure(events=nevents_cycle)
    dets = [daq]
    _set_bykik(bykik)
    
    if (motor is not None):
        daq.configure(motors=[motor])
     
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
    
    def step(daq, motor, start, stop, nstep, nevents_step, record=True, bykik=-1, restore=True):
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
        _set_bykik(bykik)      

        #Scan
        RE(bp.scan(dets, motor, start, stop, nstep))
        daq.control.setRecord(False)
                    
        if (record): #Post scan details to elog. 
            msg = "Scan " + motor.name + " from " + str(start) + " to " + str(stop) + " in " + str(nstep) + " steps."    
            elog.post(msg,run = daq.control.getStatus()[-1])
        print('\a')
        if (restore):
            motor.mv(start_position) #Return to start position

    @staticmethod
    def list(daq, motor, posList, nevents_step, record=True, bykik=-1, restore=True):
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
        _set_bykik(bykik)      

        #Scan
        RE(bp.list_scan(dets, motor, posList))
        daq.control.setRecord(False)
        #Post scan details to elog.
        if (record):  
            msg = "Scan " + motor.name + ": " + str( np.sort( np.unique( posList ) ) )
            elog.post(msg,run = daq.control.getStatus()[-1])
        print('\a')
        if (restore):
            motor.mv(start_position)

    def random(daq, motor, posList, nevents_step, tot_events, record=True, bykik=-1, restore=True):
        """
        Random scan of a motor. This function will randomly scan the "motor" through the "posList" taking "nevents_step" per step until it has completed at least "tot_events".
        bykik: set bykik to abort every N shots. bykik = 0 will stop bykik, bykik = -1 will not change the current setting. 
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

    def zigzag(daq, motor, posList, nevents_step, tot_events, record=True, bykik=-1, restore=True):
        """
        zig-zag (or snake) scan of a motor. This function will scan the "motor" through the "posList" in a zig-zag fashion taking "nevents_step" per step until it has completed at least "tot_events".
        bykik: set bykik to abort every N shots. bykik = 0 will stop bykik, bykik = -1 will not change the current setting. 
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

    def random_ordered(daq, random_motor, random_list, ordered_motor, ordered_list, nevents_step, tot_events, record=True, bykik=-1, restore=True):
        """
        Random scan of one motor and snake scan of another. This function will randomly scan the "random_motor" through the "random_list" while also scanning through the "ordered_list" with the "ordered_motor," taking "nevents_step" per step until it has completed at least "tot_events".
        bykik: set bykik to abort every N shots. bykik = 0 will stop bykik, bykik = -1 will not change the current setting. 
        restore: returns the motor to the starting position after the scan
        """
        if isinstance(random_list, np.ndarray):
            random_list = random_list.tolist()
        if not isinstance(random_list, list):
            print('you need to lass a list of steps')
            return

        if isinstance(ordered_list, np.ndarray):
            ordered_list = ordered_list.tolist()
        if not isinstance(ordered_list, list):
            print('you need to lass a list of steps')
            return

        #Build Scan Lists
        random_List = []
        ordered_List = []
        num_data = 0
        ordered_list.sort()
        while num_data < tot_events:

            for ele in ordered_list:
                ordered_List.extend( [ele] * len(random_list) )
                np.random.shuffle(random_list)
                random_List.extend(random_list)    
            
                num_data += nevents_step * len(random_list)
            ordered_list.reverse()
        
        #setup parameters for scan
        dets = [daq]
        daq.configure(motors = [random_motor, ordered_motor])     
        daq.configure(record=record)        
        #daq.readoutCount = nevents_step
        daq.configure(events=nevents_step)
        start_position1 = random_motor.position
        start_position2 = ordered_motor.position

        #bykik
        _set_bykik(bykik)
    
        #execute scan
        RE( bp.list_scan(dets, random_motor, random_List, ordered_motor, ordered_List) )
        if (record): #Post scan details to elog. 
            msg = "Scan " + random_motor.name + ": " + str(random_list) + " " + ordered_motor.name + ": " + str(ordered_list)    
            elog.post(msg,run = daq.control.getStatus()[-1]) 
        daq.control.setRecord(False)

        #return motors to start positions
        if (restore):
            random_motor.mv(start_position1)
            ordered_motor.mv(start_position2)

