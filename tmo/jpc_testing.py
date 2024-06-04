from tmo.db import RE, bp, bps, bpp
from pcdsdevices import utils
from tmo.bykiks import set_bykik
from collections import defaultdict
import threading 

def fly(daq, motor, start, nevents_cycle=120, record=True, bykik=-1):
    """
    Function for a fly scan with the DAQ.
    Arguments:
    motor: motor to fly scan.
    nevents_cycle: Number of events between screen updates, default is 1 sec at 120 Hz (120 events). 
    record: Boolean determining recording of the run.
    bykik: set bykik to abort every N shots. bykik = 0 will stop bykik. bykik = -1 will not change the current status. 
    """           
    daq.control.setState('connected')
    daq.configure(record=record)        
    daq.configure(events=nevents_cycle)
    daq.configure(motors=[motor])
    dets = [daq]
    set_bykik(bykik)
    
    RE(fly_scan(dets, motor, start, num=nevents_cycle))

def fly_scan(detectors, motor, start, num=1):

    md = {'detectors': [det.name for det in detectors],
           'motors': [motor.name],
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
                        next_pos = float(u_input)
                        done = False
                    elif u_input == 'q':
                        done = True

        threading.Thread(target = thread_keyboard).start()

        while not done:
            pos_cache = defaultdict(lambda: None)
            step = {motor:next_pos}
            yield from bps.one_nd_step(detectors, step, pos_cache)
            #yield from bps.mv(motor, next_pos)
#            u_input = input("Next point (q to quit): ")
#            if u_input == '':
#                next_pos = next_pos
#                done = False
#            elif is_number(u_input):
#                next_pos = float(u_input)
#                done = False
#            elif u_input == 'q':
#                done = True

         
    return (yield from fly_core())
       
        
        
