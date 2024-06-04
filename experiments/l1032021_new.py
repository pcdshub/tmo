import logging
import numpy as np
import time
from tmo.bykiks import bykiks
from tmo.db import motors, RE, bp
from elog import HutchELog as elog_temp
elog = elog_temp.from_registry()
logger = logging.getLogger(__name__)

def x_scan(daq, x_motor, start, step, nstep, nevents_step, record=True, restore=True, y_motor=None, book_end=50):
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


