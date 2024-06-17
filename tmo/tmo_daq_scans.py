import numpy as np
import time as time
#from elog import HutchELog as elog_temp
#elog = elog_temp.from_registry()
#import pandas as pd
from tmo.bykiks import bykiks, set_bykik
from tmo.db import RE, bp, bps, bpp
from ophyd.device import Component as Cpt
from ophyd import EpicsSignalRO
from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from pcdsdevices.sim import FastMotor
from collections import defaultdict
import threading 
import os
#from tmo.tmo_daq_tmp import get_daq
from tmo.db import daq, elog

sim_motor = FastMotor(name='DAQ Motor')

def _BuildList(input_list, scan_type, nevents_step, tot_events, max_events_run, repeat=1):

    #Check Input
    if isinstance(input_list, np.ndarray):
        input_list = input_list.tolist()
    if not isinstance(input_list, list):
        print('you need to pass a list of steps')
        return None
    if not(scan_type in ['r','z']):
        print('invalid scan type, must be (r)andom or (z)ig-zag')
        return None

    #Build Scan Lists
    output_list = []
    num_data = 0
    if scan_type == 'r':
        np.random.shuffle(input_list)
    elif scan_type == 'z':
        input_list.sort()
        
    while num_data < tot_events:
        for ele in input_list:
            output_list.extend( [ele] * repeat )
            num_data += nevents_step * repeat
            
        if scan_type == 'r': 
            np.random.shuffle(input_list)
        elif scan_type == 'z':
            input_list.reverse()

    #Reshape lists based on runs
    if (max_events_run > 0):
        num_steps_tot = tot_events/nevents_step
        num_steps_run = np.ceil(max_events_run/nevents_step).astype(int)
        num_runs = np.floor(num_steps_tot/num_steps_run).astype(int)
        num = num_runs * num_steps_run
        tempList = np.reshape(output_list[0:num], (num_runs,num_steps_run))
        tempList = tempList.tolist()
        if (len(output_list) > num ):
            tempList.append(output_list[num:])
        output_list = tempList
#    else:
#        output_list = [output_list]
       
    return output_list 

class DaqScan():
    """
    Class of scans involving the DAQ. This is really just a wrapper for bluesky function with some useful list generators. The main workhorse is bluesky.list_scan
    0D scans:
        untimedRun: Start the DAQ in a free running mode. 
        timedRun: Put the DAQ in a running state and start a timer, when the timer expires, the DAQ wil stop running. Likely depracated. 
        timedRun2: Similar to timedRun, but the time is measured in the number of events, not wall time. 
    1D scans:
        lists: Scan a motor through a list of points.
        step: Scan a motor by stepping between a starting an stopping point.
        zigzag: zig-zag (or snake) scan of a motor through a list of points.
        random: randomize the order of a list and scan the motor through a list of points. The list is randomized every scan.
        fly: Attempt to stabilize fly scans of a motor. The DAQ will interput occasionally to try to synchronize steps in a fly scan.
        adaptive: work in progress
    2D scans:
        xy_scan: scan two motors through a 2D space 
        zigzag_dual: scan two motors together through a list of points
    
    Tools:
        scan.runnum(): return the most recent run number (either active or complete). 
        scan.stopRun(): Stops the current run.
        scan.log_scan(): post a message about the current scan (most recent run number) to the elog. 
    
    Attributes:
        daq: daq object
        nevents_step: number of events to take at asingle point in the scan.
        tot_events: Total number of events to save for this acquisition. Could be split into multiple runs based on max_events_run
        record: Boolean Flag to record the run
        max_events_run: maximum number of events to have for a single run. Not all scans use this. 
        
    Example: full function definition:
        scan.zigzag(motor1, list1, nevents_step=8000*5, tot_events=8000*5*len(list1)*5, record=True)
                    
    Example: using attributes of scan:
        The properties of the scan can be stored as attributes of the class so they do not need to be redefinecd every scan. 
        >>list1 = [points to scan thru]
        >>scan.nevents_step = 8000*5 # 5 sec @ 8 kHz
        >>scan.tot_events = 8000*5*len(list1)
        >>scan.record = True
        >>scan.zigzag(motor1, list1)
        >>scan.random(motor1, list1)
        
    """
    def __init__(self, daq):
        self.daq = daq
        self.nevents_step=1E4 
        self.tot_events=1E5
        self.record = True
        self.max_events_run=0
        
    def _update_scan_param(self, nevents_step, tot_events, max_events_run, record):
        self.nevents_step = nevents_step if nevents_step is not None else self.nevents_step
        self.tot_events = tot_events if tot_events is not None else self.tot_events
        self.max_events_run = max_events_run if max_events_run is not None else self.max_events_run
        self.record = record if record is not None else self.record
        
    def log_scan(self, msg):
        """
        Post a message about the current scan to the elog

        Parameters
        ----------
        msg : str
            Description of scan.

        Returns
        -------
        None.

        """
        #run_num = np.max(self.daq.control.getStatus()[-2:])
        elog.post(msg,run=self.run_num(), tags='daq_scan')   

    def run_num(self):
        runs = [item for item in daq.control.getStatus() if (isinstance(item, int) and not isinstance(item, bool))]
        if len(runs) == 0:
            run_num=None
        else:
            run_num = np.max(runs)
        return run_num
    
    def _setup_freeRun(self, record=False):
        self.daq.unstage()
        self.daq._set_connected() #This should be redundant.
        with self.daq.daqState_cv:
            while self.daq.daqState != 'connected':
                self.daq.daqState_cv.wait(1.0)
            self.daq.configure(events=-1, record=record)
        self.daq.stage()
    
    def _timedRun(self, duration=10, record=True):
        self.daq.control.setState('configured')
        prev_rec_state = self.daq.control.getStatus()[3] #attempt to make changing record state more stable        
        self.daq.control.setRecord(record)
        self.daq.control.setState('running')
        print('record = ', self.daq.control.getStatus()[3])
        time.sleep(duration)
        self.daq.control.setState('configured')
        self.daq.control.setRecord(False)

    def timedRun(self, duration=10, nrep=1, record=None, freeRun=False):
        """
        Function starts a series of DAQ runs with a time limit.
        Parameters:
        -----------
        duration: float, Duration of individual DAQ run [s]
    
        nrep: int, 
            Number of individual DAQ runs
    
        record: bool, 
            Flag to tell DAQ to save the run.
         
        freeRun: bool, 
            Start untimedRun after the timedRun is complete
        """
        self._update_scan_param(None, None, None, record)
        try:
            for irep in range(nrep):
                self._timedRun(duration, self.record)
            print('Timed run has finished')
            os.system('echo -e "\a"');
            if freeRun:
                self.untimedRun(False)
        except KeyboardInterrupt:
            self.daq.control.setState('configured')
              
    def untimedRun(self, record=None):
        """
        Function starts a DAQ run without a time limit.

        Parameters:
        -----------
        record: bool, 
            Flag to tell DAQ to save the run.

        """
        self._update_scan_param(None, None, None, record)
        self._setup_freeRun(record=self.record)
        self.daq.control.setState('running')

        with self.daq.daqState_cv:
            while self.daq.daqState != 'running':
                self.daq.daqState_cv.wait(1.0)
        print('daq.control.setState("configured") to stop run')
        
    def stopRun(self, record=False):
        """
        This function stops the current run and sets the record state.
        Parameters:
        ---------- 
        record: bool, 
            Flag to tell DAQ to save the run. 
        """
        self.daq._set_connected()
        self.daq.configure(record=record)
        self.daq.control.setRecord(record)
       
    def timedRun2(self, tot_events=None, nevents_cycle=None, num_runs=1, record=None, freeRun=False, motor=None):		
        """
        Function starts a series of DAQ runs with a time limit.
        Parameters:
        -----------
        nevents_cycle: int, 
            Number of events between screen updates, default is 30 sec at 120 Hz (3600 events).
     
        tot_events: int, 
            Total number of events in a run. 
        
        num_runs: int, 
            Number of runs to take, each run will have tot_events
        
        record: bool, 
            Flag to tell DAQ to save the run.
    
        freeRun: bool,
            Start untimedRun after the timedRun is complete
    
        motor: optional input of a motor name to append to the data
        """           
        self.daq.control.setState('connected')
        self._update_scan_param(nevents_cycle, tot_events, None, record)
        dets = [self.daq]
        
        if (motor is not None):
            if isinstance(motor, list):
                self.daq.configure(motors=motor, record=self.record, events=self.nevents_step)
            else:
                self.daq.configure(motors=[motor], record=self.record, events=self.nevents_step)
        else:
            self.daq.configure(record=self.record, events=self.nevents_step)
            
        if ( self.nevents_step > self.tot_events):
            print("Error: Requested more events/cycle than total events")
            return
        num_cycles = np.ceil(self.tot_events/self.nevents_step).astype(int);
        #The run engine should not go inside the for loop. Instead the loop should be executed as below then the RE called once
        def temp_plan(num_runs, num_cycles):
            for irep in range(num_runs):
                yield from bp.count(dets, num=num_cycles)
        RE(temp_plan(num_runs, num_cycles))
        os.system('echo -e "\a"');
#        self._setup_freeRun(record=False)
        if freeRun:
            self.untimedRun(False)    
    
    def step(self, motor, start, stop, nstep, nevents_step=None, record=None, restore=True):
        """
        Perform a step scan of a motor
        Parameters:
        -----------
        motor: Device to scan
    
        start: float, 
            Starting position of the scan
    
        stop: float, 
            Last position in the scan
    
        nstep: float, 
            Number of steps to take between start and stop
    
        nevents_step: int, 
            Number of events to record at each step
    
        record: bool, 
            Flag to tell DAQ to save the run. 
    
        restore: bool, 
            Returns the motor to the starting position after the scan
        """
        self.daq.control.setState('connected')
        self._update_scan_param(nevents_step, None, None, record)
        #Setup DAQ and scan parameters
        dets = [self.daq]
        self.daq.configure(events=self.nevents_step, record=self.record, motors=[motor])
        start_position = motor.position
            
        #Scan
        RE(bp.scan(dets, motor, start, stop, nstep))
                        
        if (self.record): #Post scan details to elog. 
            msg = "Scan " + motor.name + " from " + str(start) + " to " + str(stop) + " in " + str(nstep) + " steps."    
            self.log_scan(msg)
        os.system('echo -e "\a"');
        if (restore):
            motor.move(start_position) #Return to start position
            
    def lists(self, motor, posList, nevents_step=None, record=None, restore=True):
        """
        Scan "motor" through a list of points. 
        Parameters:
        ----------
        motor: Device to scan
    
        posList: list of float, 
            List of positions to visit. If posList is a list of lists, then each list will be a seperate run. 
    
        nevents_step: int, 
            Number of events to take at each step.
        
        record: bool, 
            Flag to tell DAQ to save the run. 
    
        restore: bool, 
            Returns the motor to the starting position after the scan
        """
        self.daq.control.setState('connected')
        self._update_scan_param(nevents_step, None, None, record)    
        #Setup DAQ and scan parameters
        if isinstance(posList, np.ndarray):
            posList = posList.tolist()
        if not isinstance(posList, list):
            print('you need to pass a list of steps')
            return      
        dets = [self.daq]
        self.daq.configure(events=self.nevents_step, record=self.record, motors=[motor])
        start_position = motor.position
    
        #Scan the list
        if ( any( isinstance(el, list) for el in posList ) ): #posList is list of lists. Need multiple runs
            def temp_plan(posList): #define plan to iterate over runs
                for lst in posList:
                    yield from bp.list_scan(dets, motor, lst)
                    if (self.record):
                        msg = "Scan " + motor.name + ": " + str( np.sort( np.unique( lst ) ) )
                        self.log_scan(msg)
            RE( temp_plan(posList) )
                
        else: #posList is a list of points to visit 
            RE(bp.list_scan(dets, motor, posList))
            #Post scan details to elog.
            if (self.record):  
                msg = "Scan " + motor.name + ": " + str( np.sort( np.unique( posList ) ) )
                self.log_scan(msg)
        os.system('echo -e "\a"');
        if (restore):
            motor.move(start_position)
    
    def random(self, motor, posList, nevents_step=None, tot_events=None, record=None, restore=True, max_events_run=None):
        """
        Random scan of a motor. This function will randomly scan the "motor" through the "posList" taking "nevents_step" per step until it has completed at least "tot_events".
        Parameters:
        -----------
        motor: Device to scan
    
        posList: List of float, 
            List of positions to visit, will be resorted randomly each iteration
    
        nevents_step: int, 
            Number of events to take at each position. 
    
        tot_events: int, 
            Total number of events to take
    
        record: bool, 
            Flag to tell DAQ to save the run. 
    
        restore: bool, 
            Returns the motor to the starting position after the scan
    
        max_events_run: int, 
            maximum number of events to have in a run. Requesting more events will split scan into multiple runs
        """
        self._update_scan_param(nevents_step, tot_events, max_events_run, record)
        #Build List
        fullList = _BuildList(posList, 'r', self.nevents_step, self.tot_events, self.max_events_run, repeat=1 )
        if fullList is None:
            return
        #use list_scan to scan this list. 
        self.lists(motor, fullList, self.nevents_step, self.record, restore) 
    
    def zigzag(self, motor, posList, nevents_step=None, tot_events=None, record=None, restore=True, max_events_run=None):
        """
        zig-zag (or snake) scan of a motor. This function will scan the "motor" through the "posList" in a zig-zag fashion taking "nevents_step" per step until it has completed at least "tot_events".
        Parameters:
        -----------
        motor: Device to scan
    
        posList: List of floats, 
            List of positions to visit, will be zig-zaged each iteration
    
        nevents_step: int, 
            Number of events to take at each position. 
    
        tot_events: int, 
            Total number of events to take
    
        record: bool, 
            Flag to tell DAQ to save the run. 
    
        restore: bool, 
            Returns the motor to the starting position after the scan
    
        max_events_run: int,
            Maximum number of events to have in a run. Requesting more events will split scan into multiple runs.
        """
        self._update_scan_param(nevents_step, tot_events, max_events_run, record)
        #Build List
        fullList = _BuildList(posList, 'z', self.nevents_step, self.tot_events, self.max_events_run, repeat=1 )
        if fullList is None:
            return                  
        #use list_scan to scan this list. 
        self.lists(motor, fullList, self.nevents_step, self.record, restore) 
    
    def lists_dual(self, x_motor, x_list, y_motor, y_list, nevents_step=None, record=None, restore=True):
        """
        Scan two motors (x and y) through a list of points (x_list and y_list). 
        Parameters:
        -----------
        x_motor: First motor to scan
    
        x_list: List of float, 
            List of positions to visit with x_motor. 
    
        y_motor: Second motor to scan. 
    
        y_list: List of float, 
            List of positions to visit with y_motor. 
    
        nevents_step: int, 
            Number of events to take at each position. 
        
        record: bool, 
            Flag to tell DAQ to save the run. 
            
        restore: bool, 
            Returns the motor to the starting position after the scan
        """
        if isinstance(x_list, np.ndarray):
            x_list = [x_list.tolist()]
        if not isinstance(x_list, list):
            print('you need to pass a list of steps for x_motor')
            return  
        if isinstance(y_list, np.ndarray):
            y_list = [y_list.tolist()]
        if not isinstance(y_list, list):
            print('you need to pass a list of steps for y_motor')
            return  
        if not isinstance(x_list[0], list):
            x_list=[x_list]
        if not isinstance(y_list[0], list):
            y_list=[y_list]
        if len(x_list[0]) != len(y_list[0]):
            print('x_list and y_list must have the same length.')
            return
        self.daq.control.setState('connected')
        self._update_scan_param(nevents_step, None, None, record)                                
        #setup parameters for scan
        dets = [self.daq]
        self.daq.configure(motors = [x_motor, y_motor], record=self.record, events=self.nevents_step)     
        start_position_x = x_motor.position
        start_position_y = y_motor.position
            
        #execute scan
        def temp_plan(x_List, y_List): #define plan to iterate over runs
            for i, (x_lst, y_lst) in enumerate( zip( x_List, y_List) ):
                yield from bp.list_scan(dets, x_motor, x_lst, y_motor, y_lst)
                if (self.record):
                    msg = "Scan "+x_motor.name+": "+ str(np.sort(np.unique(x_lst)))+"\n"+y_motor.name+": "+str(np.sort(np.unique(y_lst)))
                    self.log_scan(msg)
        RE( temp_plan(x_list, y_list) )  
        self.daq.control.setRecord(False)
    
        #return motors to start positions
        if (restore):
            x_motor.move(start_position_x)
            y_motor.move(start_position_y)

    def xy_scan(self, x_motor, x_list, y_motor, y_list, nevents_step=None, tot_events=None, record=None, scan_type =['r','z'], restore=True, max_events_run=None):
        """
        Scan two motors (x and y). This function will scan the "x_motor" through the "x_list" while also scanning through the "y_list" with the "y_motor," taking "nevents_step" per step until it has completed at least "tot_events".
        The 'scan_type' input specifies wether the motor is scanned (r)andomly or (z)ig-zagged. The default is to scan x-motor randomly and y-motor zig-zag.  
        Parameters:
        -----------
        x_motor: First motor to scan
    
        x_list: List of float, 
            List of positions to visit with x_motor. 
    
        y_motor: Second motor to scan. 
    
        y_list: List of float, 
            List of positions to visit with y_motor. 
    
        nevents_step: int, 
            Number of events to take at each position. 
    
        tot_events: int, 
            Total number of events to take
        
        record: bool, 
            Flag to tell DAQ to save the run. 
    
        scan_type: list of str,
            List of the scan type for each axis. The first element describes the x-axis, the second elements describes the y-axis.
            r - for random scan
            z - for zig-zag or snake scan
        
        restore: bool, 
            Returns the motor to the starting position after the scan
    
        max_events_run: int, 
            Maximum number of events to have in a run. Requesting more events will split scan into multiple runs
        """
        self._update_scan_param(nevents_step, tot_events, max_events_run, record)
        #Build Lists
        y_List = _BuildList(y_list, scan_type[1], self.nevents_step, self.tot_events, self.max_events_run, repeat=len(x_list))
        if not isinstance(y_List[0], list):
            y_List = [y_List]
        x_List = _BuildList(x_list, scan_type[0], self.nevents_step, sum([len(ele) for ele in y_List])*self.nevents_step, self.max_events_run, repeat=1)
        if not isinstance(x_List[0], list):
            x_List = [x_List]
        if ( (y_List[0] is None) or (x_List[0] is None) ):
            return

        self.lists_dual(x_motor, x_List, y_motor, y_List, self.nevents_step, self.record, restore)
    
    def zigzag_dual(self, x_motor, x_list, y_motor, y_list, nevents_step=None, tot_events=None, record=None, restore=True, max_events_run=None):
        """
        Scan two motors (x and y). This function will scan the "x_motor" through the "x_list" while also scanning through the "y_list" with the "y_motor," taking "nevents_step" per step until it has completed at least "tot_events".
        
        Parameters:
        -----------
        x_motor: First motor to scan
    
        x_list: List of float, 
            List of positions to visit with x_motor. 
    
        y_motor: Second motor to scan. 
    
        y_list: List of float, 
            List of positions to visit with y_motor. 
    
        nevents_step: int, 
            Number of events to take at each position. 
    
        tot_events: int, 
            Total number of events to take
    
        record: bool, 
            Flag to tell DAQ to save the run. 
    
        restore: bool, 
            Returns the motor to the starting position after the scan
    
        max_events_run: int, 
            Maximum number of events to have in a run. Requesting more events will split scan into multiple runs
        """
        self._update_scan_param(nevents_step, tot_events, max_events_run, record)
        #Build Lists
        if len(x_list) != len(y_list):
            print('x_list and y_list must have the same length.')
            return
        y_List = _BuildList(y_list, 'z', self.nevents_step, self.tot_events, self.max_events_run, repeat=1 )
        if not isinstance(y_List[0], list):
            y_List = [y_List] 
        x_List = _BuildList(x_list, 'z', self.nevents_step, self.tot_events, self.max_events_run, repeat=1 )
        if not isinstance(x_List[0], list):
            x_List = [x_List] 
        if ( (y_List[0] is None) or (x_List[0] is None) ):
            return
        
        self.lists_dual(x_motor, x_List, y_motor, y_List, self.nevents_step, self.record, restore)  
           
    def fly(self, motor, start, limits=[-1,1], nevents_cycle=None, record=None):
        """
        "Fly" scan with the DAQ.
        Parameters:
        -----------
        motor: motor to fly scan.
    
        start: float,
            Starting position of motor.
    
        limits: list of floats,
            Limits for fly scan. If the requested position is outside the limits, no step is made
    
        nevents_cycle: int, 
            Number of events between screen updates, default is 1 sec at 120 Hz (120 events). 
    
        record: bool,
            Flag to tell DAQ to save the run.
    
        """           
        self._update_scan_param(nevents_cycle, None, None, record)
        self.daq.control.setState('connected')
        self.daq.configure(record=self.record, events=self.nevents_step, motors=[motor])
        dets = [self.daq]
        
        RE(self._fly_scan(dets, motor, start, limits, num=self.nevents_step))
    
    def _fly_scan(self, detectors, motor, start, limits, num=1):
    
        md = {'detectors': [det.name for det in detectors],
               'motors': [motor.name],
               'extents': tuple([min(limits), max(limits)]),
               'plan_args': {'detectors': list(map(repr, detectors)),
                             'motor': repr(motor)},
               'plan_name': 'fly_scan',
               'hints': {},
               }
    
        @bpp.stage_decorator(list(detectors) + [motor])
        @bpp.run_decorator(md=md)
        def fly_core():
            global next_pos
            global done
    
            next_pos = start
            done = False
            
            def is_number(n):
                try:
                    float(n)
                    return True
                except ValueError:
                    return False         
        
            def thread_keyboard():
                global next_pos
                global done
                done = False
                next_pos = start
                lock = threading.Lock()
                while not done:
                    with lock:
                        u_input = input("Next point (q to quit): ")
                        if is_number(u_input):
                            if (min(limits) <= float(u_input) <= max(limits)):
                                next_pos = float(u_input)
                            done = False
                        elif u_input == 'q':
                            done = True
    
            threading.Thread(target = thread_keyboard).start()
    
            while not done:
                pos_cache = defaultdict(lambda: None)
                step = {motor:next_pos}
                yield from bps.one_nd_step(detectors, step, pos_cache)
    
             
        return (yield from fly_core())
    
    def adaptive_step_find(self, feedback_device, motor, start, limits, step_level, threshold_step,  nevents_cycle=None, record=None):
        """
        Scan motor to find a step in the feedback device
        Parameters:
        -----------
        feedback_device: Detector Object to use as feedback in the scan
    
        motor: motor to scan.
    
        start: float,
            Starting position of motor.
    
        limits: list of floats,
            Limits for the search range. 
    
        step_level: float,
            Value to differentiate between low side and high side of step.
    
        thershold_step: float,
            Fidelity to determine step position with the motor.
    
        nevents_cycle: int, 
            Number of events per step, default is 1 sec at 120 Hz (120 events). 
    
        record: bool,
            Flag to tell DAQ to save the run.

        """
        self._update_scan_param(nevents_cycle, None, None, record)       
        #Setup DAQ
        self.daq.control.setState('connected')
        self.daq.configure(record=record, events=self.nevents_step, motors=[motor])
        dets = [self.daq, feedback_device]
    
    #def _adaptive_step_find_scan(detectors, motor, start, limits, thresholds, num=1):

#from tmo.db import daq
scan = DaqScan(daq)

class DaqStats(BaseInterface, Device):
    tab_component_names = True
    
    dead_frac = Cpt(EpicsSignalRO, ':DeadFrac')
    damage_hsd = Cpt(EpicsSignalRO, ':Damage_hsd')
    damage_bld = Cpt(EpicsSignalRO, ':Damage_bld')
    damage_epics = Cpt(EpicsSignalRO, ':Damage_epics')
    damage_atm = Cpt(EpicsSignalRO, ':Damage_atm')
    damage_fzp = Cpt(EpicsSignalRO, ':Damage_fzp')
    
daq_stats = DaqStats('DAQ:NEH:tmo:0',name='damage_stats')