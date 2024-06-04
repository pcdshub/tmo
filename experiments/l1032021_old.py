import logging
import numpy as np
import time
from tmo.bykiks import bykiks
from tmo.db import motors, RE, bp
from elog import HutchELog as elog_temp
elog = elog_temp.from_registry()
logger = logging.getLogger(__name__)

def x_scan(daq, x_start, x_step, num_steps, num_events_step, num_events_fiducal):

    if not bykiks.burst_mode_enable.get():
        logger.warning('Burst Mode not enabled, call ACR.  Exiting')
        return

    #assign x and y motors
    motor_x = motors.tmo_lamp_magnet_x
    motor_y = motors.tmo_lamp_magnet_y

    #setup DAQ
    daq.control.setState('configured')
    prev_rec_state = daq.control.getStatus()[3] #attempt to make changing record state more stable        
    daq.control.setRecord(True)
    daq.control.setState('running')
    logger.info('record = ', daq.control.getStatus()[3])

    #move to starting position
    motor_x.mv(x_start, wait=True)    
    step = 0

    #Mark Fidicuial
    bykiks.shots_to_burst.put(num_events_fiducal)
    logger.info('Marking Fiducial at ',str(motor_x.position))
    bykiks.request_burst.put(1)
    while bykiks.request_burst.get():
        sleep(0.1)
    
    bykiks.shots_to_burst.put(num_events_step)
    while step<num_steps:
        step += 1
        motor_x.mvr(x_step, wait=True)
        logger.info('Collecting Data at: ', str(motor_x.position))
        #burst
        bykiks.request_burst.put(1)
        while bykiks.request_burst.get():
            sleep(0.1)       

    daq.control.setState('configured')
    msg = "Scan " + motor_x.name + " from " + str(x_start) + " in " + str(num_steps) + " steps of " + str(x_step) + " taking " + str(num_events_step) + " events each step. " + motor_y.name + " at " + str(motor_y.position)
    elog.post(msg,run = daq.control.getStatus()[-1])       

    motor_x.mv(x_start, wait=True)

def x_scan_bp(daq, x_motor, start, step, nstep, nevents_step, record=True, restore=True, y_motor=None):
    """
        Scan "x_motor" from "start" position with "nstep" steps of size "step."
        nevents_step: number of events to take at each step.
        record: sets the record mode for the DAQ
        restore: returns the motor to the starting position after the scan
        y_motor: optional input to include the y_motor position in elog posts.
    """
    if not bykiks.burst_mode_enable.get():
        logger.warning('Burst Mode not enabled, call ACR.  Exiting')
        return

    #build list of points
    posList = [];
    n_step = 0
    while n_step < nstep:
        posList.append(start + n_step*step)
        n_step += 1 
    
    #setup parameters for scan
    dets = [daq, bykiks]
    daq.configure(motors = [x_motor])     
    daq.configure(record=record)        
    daq.configure(events=nevents_step)

    bykiks.shots_to_burst.put(nevents_step)
    start_position = x_motor.position    
    daq.control.setState('connected')

    #Run the Scan
    RE(bp.list_scan(dets, x_motor, posList))
    
    #Post scan details to elog.
    if (record):  
        msg = "Scan " + x_motor.name + ": " + str( np.sort( np.unique( posList ) ) )
        if y_motor is not None:
            msg = msg + "\n" + y_motor.name + ": " + str( y_motor.position ) 
        elog.post(msg,run = daq.control.getStatus()[-1])
       
    daq.control.setRecord(False)
    if (restore):
        x_motor.move(start_position)





