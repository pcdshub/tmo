from elog import HutchELog as elog_temp
elog = elog_temp.from_registry()
from tmo.db import bp, RE
from tmo.bykiks import bykiks, set_bykik

import pandas as pd
import numpy as np
import time as time

    
class User:    
    
    def twoD_zig_zag(self, daq, motor_1, motor_1_list, motor_2, motor_2_list, nevents_step, tot_events, record=True, bykik=-1, restore=True, max_events_run=0):
        """
        Snake scan two motors. This function will 2-D snake scan the "motor_1" through the "motor_1_list" while also scanning through the "motor_2_list" with the "motor_2," taking "nevents_step" per step until it has completed at least "tot_events".
        bykik: set bykik to abort every N shots. bykik = 0 will stop bykik, bykik = -1 will not change the current setting. 
        restore: returns the motor to the starting position after the scan
        max_events_run is the maximum number of events in a run. Having tot_events > max_events_run will partion into several runs. 
        """
        if isinstance(motor_1_list, np.ndarray):
            motor_1_list = motor_1_list.tolist()
        if not isinstance(motor_1_list, list):
            print('you need to lass a list of steps')
            return

        if isinstance(motor_2_list, np.ndarray):
            motor_2_list = motor_2_list.tolist()
        if not isinstance(motor_2_list, list):
            print('you need to lass a list of steps')
            return
 
        #Build Scan Lists
        motor_1_List = [] #long list of all the points which will be visited
        motor_2_List = [] 
        num_data = 0
        motor_1_list.sort() #short list 
        motor_2_list.sort()
        
        while num_data < tot_events:

            for ele in motor_2_list:
                motor_2_List.extend( [ele] * len(motor_1_list) )
                motor_1_List.extend(motor_1_list)
                motor_1_list.reverse()    
            
                num_data += nevents_step * len(motor_1_list)
            motor_2_list.reverse()

	#Reshape lists based on number of runs
        if (max_events_run > 0):
            num_steps_tot = tot_events/nevents_step
            num_steps_run = np.ceil(max_events_run/nevents_step).astype(int)
            num_runs = np.floor(num_steps_tot/num_steps_run).astype(int)
            num = num_runs * num_steps_run
            tempList1 = np.reshape(motor_1_List[0:num], (num_runs,num_steps_run))
            tempList1 = tempList1.tolist()
            tempList2 = np.reshape(motor_2_List[0:num], (num_runs,num_steps_run))
            tempList2 = tempList2.tolist()
            if (len(motor_1_List) > num ):
                tempList1.append(motor_1_List[num:])
                tempList2.append(motor_2_List[num:])
            motor_1_List = tempList1
            motor_2_List = tempList2
        else:
            motor_1_List = [motor_1_List]
            motor_2_List = [motor_2_List]         

        #setup parameters for scan
        dets = [daq]
        daq.configure(motors = [motor_1, motor_2])     
        daq.configure(record=record)        
        #daq.readoutCount = nevents_step
        daq.configure(events=nevents_step)
        start_position1 = motor_1.position
        start_position2 = motor_2.position

        #bykik
        set_bykik(bykik)
    
        #execute scan
        def temp_plan(motor_1_List, motor_2_List): #define plan to iterate over runs
            for i, (r_lst, o_lst) in enumerate( zip( motor_1_List, motor_2_List) ):
                yield from bp.list_scan(dets, motor_1, r_lst, motor_2, o_lst)
                if (record):
                    msg = "Scan " + motor_1.name + ": " + str(r_lst) + " " + motor_2.name + ": " + str(o_lst)
                    elog.post(msg,run = daq.control.getStatus()[-2])
        RE( temp_plan(motor_1_List, motor_2_List) )  
        daq.control.setRecord(False)

        #return motors to start positions
        if (restore):
            motor_1.move(start_position1)
            motor_2.move(start_position2)


    def dual_zig_zag(self, daq, motor_1, motor_1_list, motor_2, motor_2_list, nevents_step, tot_events, record=True, bykik=-1, restore=True, max_events_run=0):
        """
        Snake scan two motors together. Snake scan the "motor_1" through the "motor_1_list" while also scanning through the "motor_2_list" with the "motor_2," taking "nevents_step" per step until it has completed at least "tot_events".
        bykik: set bykik to abort every N shots. bykik = 0 will stop bykik, bykik = -1 will not change the current setting. 
        restore: returns the motor to the starting position after the scan
        max_events_run is the maximum number of events in a run. Having tot_events > max_events_run will partion into several runs. 
        """
        if isinstance(motor_1_list, np.ndarray):
            motor_1_list = motor_1_list.tolist()
        if not isinstance(motor_1_list, list):
            print('you need to lass a list of steps')
            return

        if isinstance(motor_2_list, np.ndarray):
            motor_2_list = motor_2_list.tolist()
        if not isinstance(motor_2_list, list):
            print('you need to lass a list of steps')
            return
        
        if np.size(motor_1_list) != np.size(motor_2_list):
            print('motor lists should have the same number of elements')
            return

        #Build Scan Lists
        motor_1_List = [] #long list of all the points which will be visited
        motor_2_List = [] 
        num_data = 0
        motor_1_list.sort() #short list 
        motor_2_list.sort()
        
        while num_data < tot_events:

            motor_1_List.extend(motor_1_list)
            motor_2_List.extend(motor_2_list)
            motor_1_list.reverse()
            motor_2_list.reverse()
            num_data += nevents_step * len(motor_1_list)

	#Reshape lists based on number of runs
        if (max_events_run > 0):
            num_steps_tot = tot_events/nevents_step
            num_steps_run = np.ceil(max_events_run/nevents_step).astype(int)
            num_runs = np.floor(num_steps_tot/num_steps_run).astype(int)
            num = num_runs * num_steps_run
            tempList1 = np.reshape(motor_1_List[0:num], (num_runs,num_steps_run))
            tempList1 = tempList1.tolist()
            tempList2 = np.reshape(motor_2_List[0:num], (num_runs,num_steps_run))
            tempList2 = tempList2.tolist()
            if (len(motor_1_List) > num ):
                tempList1.append(motor_1_List[num:])
                tempList2.append(motor_2_List[num:])
            motor_1_List = tempList1
            motor_2_List = tempList2
        else:
            motor_1_List = [motor_1_List]
            motor_2_List = [motor_2_List]         

        #setup parameters for scan
        dets = [daq]
        daq.configure(motors = [motor_1, motor_2])     
        daq.configure(record=record)        
        #daq.readoutCount = nevents_step
        daq.configure(events=nevents_step)
        start_position1 = motor_1.position
        start_position2 = motor_2.position

        #bykik
        set_bykik(bykik)
    
        #execute scan
        def temp_plan(motor_1_List, motor_2_List): #define plan to iterate over runs
            for i, (ml1, ml2) in enumerate( zip( motor_1_List, motor_2_List) ):
                yield from bp.list_scan(dets, motor_1, ml1, motor_2, ml2)
                if (record):
                    msg = "Scan " + motor_1.name + ": " + str(ml1) + " " + motor_2.name + ": " + str(ml2)
                    elog.post(msg,run = daq.control.getStatus()[-2])
        RE( temp_plan(motor_1_List, motor_2_List) )  
        daq.control.setRecord(False)

        #return motors to start positions
        if (restore):
            motor_1.move(start_position1)
            motor_2.move(start_position2)


