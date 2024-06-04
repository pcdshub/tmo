
import numpy as np
from tmo.db import bp, RE
from elog import HutchELog as elog_temp

def random_ordered_scan(daq, random_motor, random_list, ordered_motor, ordered_list, nevents_step, tot_events, record=True, bykik=0, restore=True):
        
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
    if (bykik>0):
        bykiks.abort_enable.put(1)
        bykiks.abort_every_n.put(bykik)
    elif(bykik<0): bykiks.abort_enable.put(0)        

    #execute scan
    RE( bp.list_scan(dets, random_motor, random_List, ordered_motor, ordered_List) )
    if (record): #Post scan details to elog. 
        msg = "Scan " + random_motor.name + ": " + str(random_list) + " " + ordered_motor.name + ": " + str(ordered_list)    
        elog = elog_temp.from_registry()
        elog.post(msg,run = daq.control.getStatus()[-1]) 
    daq.control.setRecord(False)

    #return motors to start positions
    if (restore):
        random_motor.mv(start_position1)
        ordered_motor.mv(start_position2)

#class User: 
