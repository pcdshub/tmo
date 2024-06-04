from hutch_python.utils import safe_load
import numpy as np
import time as time
from elog import HutchELog as elog_temp
elog = elog_temp.from_registry()
from tmo.db import bp, RE
from tmo.bykiks import bykiks, set_bykik

class User:
    
    def scan_w2w(daq, w_energy, w_list, w2_energy, w2_list, nevents_step, tot_events, record=True, bykik=-1, restore=True, max_events_run=0):

        if isinstance(w_list, np.ndarray):
            w_list = w_list.tolist()
        if not isinstance(w_list, list):
            print('you need to lass a list of steps')
            return

        if isinstance(w2_list, np.ndarray):
            w2_list = w2_list.tolist()
        if not isinstance(w2_list, list):
            print('you need to lass a list of steps')
            return


        #build the list of points to scan
        fullList1 = []
        fullList2 = []
        num_data = 0
        w_list.sort() #order the list of points sequentially
        w2_list.sort()
        while num_data < tot_events:
            fullList1.extend(w_list) 
            fullList2.extend(w2_list)   
            num_data += nevents_step * len(w_list)
            w_list.reverse()
            w2_list.reverse()
        
        #reshape list depending on max_events
        if ( max_events_run > 0 ):
            num_steps_tot = tot_events/nevents_step
            num_steps_run = np.ceil(max_events_run/nevents_step).astype(int)
            num_runs = np.floor(num_steps_tot/num_steps_run).astype(int)
            num = num_runs * num_steps_run
            tempList1 = np.reshape(fullList1[0:num], (num_runs,num_steps_run))
            tempList2 = np.reshape(fullList2[0:num], (num_runs,num_steps_run))
            tempList1 = tempList1.tolist()
            tempList1 = tempList2.tolist()
            if (len(fullList1) > num ):
                tempList1.append(fullList1[num:])
                tempList2.append(fullList2[num:])
            fullList1 = tempList1
            fullList2 = tempList2
        else:
            fullList1 = [fullList1]
            fullList2 = [fullList2]

        #Setup DAQ  
        daq.control.setState('connected')
        dets = [daq]
        daq.configure(events=nevents_step)
        daq.configure(motors=[w_energy, w2_energy])
        daq.configure(record=record)
        start_position1 = w_energy.position
        start_position2 = w2_energy.position

        #BYKIK
        set_bykik(bykik)      

        #Build a plan for the scan
        def temp_plan(fullList1, fullList2): #define plan to iterate over runs
            for i, (lst1, lst2) in enumerate( zip( fullList1, fullList2) ):
                yield from bp.list_scan(dets, w_energy, lst1, w2_energy, lst2)
                if (record):
                    msg = "Scan " + w_energy.name + ": " + str(lst1) + " " + w2_energy.name + ": " + str(lst2)
                    elog.post(msg,run = daq.control.getStatus()[-2])
        
        #Finally, we can do the scan. 
        RE( temp_plan(fullList1, fullList2) )  
        
        #Put things back to where they were
        daq.control.setRecord(False)
        print('\a') #give a beep to indicate the scan is done. 
        if (restore):
            w_energy.mv(start_position1)
            w2_energy.mv(start_position2)




