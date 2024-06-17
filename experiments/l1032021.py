import logging
import numpy as np
import time
from ophyd.status import DeviceStatus
from tmo.bykiks import bykiks, Bykiks
from tmo.db import motors, RE, bp, bps, daq
from elog import HutchELog as elog_temp
elog = elog_temp.from_registry()
logger = logging.getLogger(__name__)

class BykiksBurst(Bykiks):
    
    def configure(self, events, daq=daq):
        self._daq = daq
        self.burst_mode_enable.put(1)
        self.shots_to_burst.put(events)
        
    def trigger(self):
        with self._daq.daqState_cv:
            while self.daq.daqState != 'running':
                self.daq.daqState_cv.wait(0.1)   
        logger.info('Burst Enable')
        self.request_burst.put(1) #request burst
        return DeviceStatus(self, done=True, success=True)

bykiks_burst = BykiksBurst('', name='bykiks_burst', read_attrs=['burst_mode_enable'])

def x_scan_old(daq, x_motor, start, step, nstep, nevents_step, record=True, restore=True, y_motor=None, book_end=50):
    """
        Scan "x_motor" from "start" position with "nstep" steps of size "step."
        
        Parameters
        ----------
        daq : daq object to control DAQ
        x_motor : motor object to scan
        start : starting position of x_motor for scan
        step : size of the x_motor step
        nstep : int (or list of int), number of steps to take.   
        nevents_step: int (or list of int), number of events to take at each step. nevents_step should be the same length as nstep.
            If (nstep, nevents_step) are lists, the scan will take nstep[i] steps with nevents_step[i] events/step.
        record : bool, optional, sets the record mode for the DAQ
        restore : bool, optional, returns the motor to the starting position after the scan
        y_motor : optional input to include the y_motor position in elog posts.
        book_end : int, number of shots to start and stop the scan (not saved in DAQ).
    """
    if not bykiks.burst_mode_enable.get():
        logger.warning('Burst Mode not enabled, call ACR.  Exiting')
        return


    #setup daq parameters for scan
    dets = [daq,bykiks]
    daq.configure(motors = [x_motor])     
    daq.configure(record=record)        

    start_position = x_motor.position    
    daq.control.setState('connected')

    #Create start feature
    x_motor.mv(start, wait=True)
    bykiks.shots_to_burst.put(book_end)
    time.sleep(0.1)
    bykiks.request_burst.put(1)
    time.sleep(0.1)
    while bykiks.request_burst.get():
        time.sleep(0.01)


    #build list of points
    pos = start
    posList = [];
    
    if ( type(nstep) is list ):
        for n in range(0,len(nstep)):
            n_step = 0
            n_List = []
            while n_step < nstep[n]:
                pos += step
                n_List.append(pos)
                n_step += 1
            posList.append(n_List)

        def temp_plan(posList): #define plan to iterate over runs        
            for n, lst in enumerate(posList):
                daq.configure(events=nevents_step[n])
                yield from bps.sleep(0.1)
                bykiks.shots_to_burst.put(nevents_step[n])
                yield from bps.sleep(0.1)
                yield from bp.list_scan(dets, x_motor, lst)
#                if (record):
#                    msg = "Scan " + x_motor.name + ": " + str( np.sort( np.unique( posList ) ) ) + ', ' + str(nevents_step[n]) + ' pulses/step'
#                    if y_motor is not None:
#                        msg = msg + "\n" + y_motor.name + ": " + str( y_motor.position ) 
#                    elog.post(msg,run = daq.control.getStatus()[-2])
        #Run the Scan
        RE( temp_plan(posList) )

    else: # nstep is a number
        n_step = 0
        while n_step < nstep :
            pos+= step
            posList.append(pos)
            n_step += 1
        daq.configure(events=nevents_step)
        bykiks.shots_to_burst.put(nevents_step)
        #Run the Scan
        RE(bp.list_scan(dets, x_motor, posList))
    
        #Post scan details to elog.
#        if (record):  
#            msg = "Scan " + x_motor.name + ": " + str( np.sort( np.unique( posList ) ) ) + ', ' + str(nevents_step) + ' pulses/step'
#            if y_motor is not None:
#                msg = msg + "\n" + y_motor.name + ": " + str( y_motor.position ) 
#            elog.post(msg,run = daq.control.getStatus()[-1])
       
    daq.control.setRecord(False)

    #Create end feature
    x_motor.mvr(step, wait=True)
    bykiks.shots_to_burst.put(book_end)
    time.sleep(0.1)
    bykiks.request_burst.put(1)
    time.sleep(0.1)
    while bykiks.request_burst.get():
        time.sleep(0.01)

    if (restore):
        x_motor.mv(start_position)


def x_scan_bp(daq, x_motor, start, step, nstep, nevents_step, record=True, restore=True, y_motor=None, book_end=50):
    """
        Bluesky scan "x_motor" from "start" position with "nstep" steps of size "step."
        
        Parameters
        ----------
        daq : daq object to control DAQ
        x_motor : motor object to scan
        start : starting position of x_motor for scan
        step : size of the x_motor step
        nstep : int (or list of int), number of steps to take.   
        nevents_step: int (or list of int), number of events to take at each step. nevents_step should be the same length as nstep.
            If (nstep, nevents_step) are lists, the scan will take nstep[i] steps with nevents_step[i] events/step.
        record : bool, optional, sets the record mode for the DAQ
        restore : bool, optional, returns the motor to the starting position after the scan
        y_motor : optional input to include the y_motor position in elog posts.
        book_end : int, number of shots to start and stop the scan (not saved in DAQ).
    """

    #verify burst mode
#    if not bykiks.burst_mode_enable.get():
#        logger.warning('Burst Mode not enabled, call ACR.  Exiting')
#        return
    #configure DAQ
    dets = [daq,bykiks]
    if y_motor is not None:
        daq.configure(motors = [x_motor,y_motor])
    else:
        daq.configure(motors = [x_motor])     
    daq.configure(record=record)        
    #log starting position
    start_position = x_motor.position    
    daq.control.setState('connected')

    full_events = [book_end] + nevents_step + [book_end]
    full_steps = [1] + nstep + [1]
    if y_motor is None:
        RE(_scan(dets, [x_motor], start, step, full_steps, full_events))
    else:
        RE(_scan(dets, [x_motor,y_motor], start, step, full_steps, full_events))
    
    if (restore):
        x_motor.mv(start_position)


def _scan(detectors, motors, start, step, nstep, nevents_step):

    daq = detectors[0]
    bykik = detectors[1]
    motor = motors[0]

    #build metadata
    motor_names = tuple(motor.name for motor in motors)
    _md = {'detectors:': [det.name for det in detectors],
           'plan_args': {'detectors': list(map(repr, detectors)),
                         'num': np.sum(nstep)},
           'plan_name': 'x_scan',
           'plan_pattern': 'inner_product',
           'motors': motor_names
           }

    for det in detectors:
        yield from bps.stage(det)

    yield from bps.open_run(md=_md)
    yield from bps.mv(motor, start)    

    #wait for daq to configure....
    while not(daq.control.getState() == 'running' ):
        yield from bps.sleep(0.1)    

    for n in range(0,len(nstep)):
        n_step = 0
        while n_step < nstep[n]:
            n_step += 1
            daq.configure(events=nevents_step[n])
            bykik.shots_to_burst.put(nevents_step[n])
            yield from bps.sleep(0.2)        
            yield from bps.trigger_and_read(detectors + motors)
            yield from bps.mvr(motor, step)
                
    yield from bps.close_run()

    for det in detectors:
        yield from bps.unstage(det)

def x_scan_free(daq, x_motor, start, step, nstep, nevents_step, record=True, restore=True, y_motor=None, book_end=50):
    """
        Free scan "x_motor" from "start" position with "nstep" steps of size "step."
        
        Parameters
        ----------
        daq : daq object to control DAQ
        x_motor : motor object to scan
        start : starting position of x_motor for scan
        step : size of the x_motor step
        nstep : int (or list of int), number of steps to take.   
        nevents_step: int (or list of int), number of events to take at each step. nevents_step should be the same length as nstep.
            If (nstep, nevents_step) are lists, the scan will take nstep[i] steps with nevents_step[i] events/step.
        record : bool, optional, sets the record mode for the DAQ
        restore : bool, optional, returns the motor to the starting position after the scan
        y_motor : optional input to include the y_motor position in elog posts.
        book_end : int, number of shots to start and stop the scan (not saved in DAQ).
    """
    if not bykiks.burst_mode_enable.get():
        logger.warning('Burst Mode not enabled, call ACR.  Exiting')
        return


    #setup daq parameters for scan
    dets = [daq]
    daq.configure(motors = [x_motor])     
    daq.configure(record=record)        

    start_position = x_motor.position    
    daq.control.setState('configured')
    daq.control.getStatus()
    daq.control.setRecord(record)
    time.sleep(1)
    daq.control.setState('running')
    daq.control.getStatus()

    #Create start feature
    x_motor.mv(start, wait=True)
    time.sleep(1)
    bykiks.shots_to_burst.put(book_end)
    time.sleep(0.1)
    bykiks.request_burst.put(1)
    logger.info('Bursting ' + str(book_end))
    time.sleep(0.1)
    while bykiks.request_burst.get():
        time.sleep(0.01)
    
    if ( type(nstep) is list ):
        for n in range(0,len(nstep)):
            n_step = 0
            while n_step < nstep[n]:
                n_step += 1
                x_motor.mvr(step, wait=True)
                time.sleep(1)
                bykiks.shots_to_burst.put(nevents_step[n])
                time.sleep(0.1)
                bykiks.request_burst.put(1)
                logger.info('Bursting ' + str(nevents_step[n]))
                time.sleep(0.1)
                while bykiks.request_burst.get():
                    time.sleep(0.01)
                
                
    #Create end feature
    x_motor.mvr(step, wait=True)
    time.sleep(1)
    bykiks.shots_to_burst.put(book_end)
    time.sleep(0.1)
    bykiks.request_burst.put(1)
    logger.info('Bursting ' + str(book_end))
    time.sleep(0.1)
    while bykiks.request_burst.get():
        time.sleep(0.01)

    daq.control.setState('configured')
    daq.control.getStatus()
    daq.control.setRecord(False)

    if (restore):
        x_motor.mv(start_position)



