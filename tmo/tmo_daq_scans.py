import numpy as np
import h5py
import time as time
from tmo.db import bp, bps, bpp #, RE
from ophyd.device import Component as Cpt
from ophyd import EpicsSignalRO
from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from pcdsdevices.sim import FastMotor, SlowMotor
from pcdsdevices.pseudopos import SyncAxis
from ophyd.status import DeviceStatus, SubscriptionStatus
from collections import defaultdict
from cytoolz import partition
import pandas as pd
from collections import OrderedDict
from datetime import datetime
from tabulate import tabulate
import threading 
import os
import logging
from bluesky.callbacks import LiveTable, LivePlot, CallbackBase
from bluesky.utils import short_uid
from bluesky.plan_stubs import checkpoint, abs_set, trigger_and_read #, wait
from bluesky.protocols import Stageable, Triggerable, Readable, Locatable, Movable, Flyable, Subscribable  
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
from tmo.db import daq, elog
from tmo.tmo_beamline_control import gmd,xgmd,two_pulse_two_color
#from tmo.tmo_optics import fzp_det
from daq.scripts.tmo_sc import procmgr_hsd
from bluesky import RunEngine
RE=RunEngine({}) #use my own RunEngine with no callbacks. 
RE.pause_msg = "\nYour RunEngine is entering a paused state. These are your options for changing\nthe state of the RunEngine:\n\nscan.resume()    Resume the scan.\nscan.abort()     Perform cleanup, then kill the scan. Mark exit_stats='aborted'.\nscan.stop()      Perform cleanup, then kill the scan. Mark exit_status='success'.\nscan.halt()      Emergency Stop: Do not perform cleanup --- just stop.\n"

sim_motor = FastMotor(name='DAQ Motor')


def _BuildList(input_list, scan_type, nevents_step, tot_events, max_events_run, repeat=1):

    #Check Input
    if isinstance(input_list, np.ndarray):
        input_list = input_list.tolist()
    if not isinstance(input_list, list):
        print('you need to pass a list of steps')
        return None
    if not(scan_type in ['r','z','f']):
        print('invalid scan type, must be (r)andom, (z)ig-zag, or (f)old.')
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
        elif ( (scan_type == 'z') or (scan_type == 'f') ):
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

class DaqProfile(Device):
    """
    Device for profiling bluesky scans with the DAQ. 
    """
    def __init__(self, daq, **kwargs):
        self._daq = daq
        self.data = OrderedDict()
        self.prev_evt=0
        super().__init__(**kwargs)
        
    def stage(self):
        timestamp = time.time()
        self.data['motor_transition']={'value':0, 'timestamp':timestamp}
        self.data['daq_transition']={'value':0, 'timestamp':timestamp}
        self.data['daq_record']={'value':0, 'timestamp':timestamp}
        self.prev_evt=time.time()
        return [self]
    
    def trigger(self):
        t_0 = time.time()
        while self._daq.daqState != 'running':
            continue
        t_end = time.time()
        self.data['motor_transition']['value']=t_0-self.prev_evt
        self.data['motor_transition']['timestamp']=t_0
        self.data['daq_transition']['value']=t_end-t_0
        self.data['daq_transition']['timestamp']=t_end
        return DeviceStatus(self, done=True, success=True)
    
    def read(self):
        while self._daq.daqState == 'running':
            continue
        t_end = time.time()
        t_0 = self.data['daq_transition']['timestamp']
        self.data['daq_record']['value']=t_end-t_0
        self.data['daq_record']['timestamp']=t_end
        self.prev_evt = t_end
        return self.data 
    
    def describe(self):
        data = {'motor_transition': {'source':'', 'dtype':'number','shape':[], 'units':'s'},
                  'daq_transition': {'source':'', 'dtype':'number','shape':[], 'units':'s'},
                      'daq_record': {'source':'', 'dtype':'number','shape':[], 'units':'s'}
                }
        return data
    
    def unstage(self):
        return [self]

def one_1d_step_no_wait(detectors, motor, step):
    """
    Inner loop of a 1D step scan that does not wait for motors to finish moving.

    This is the default function for ``per_step`` param in 1D plans, except the wait is removed.
    """
    yield from checkpoint()
    yield from abs_set(motor, step, wait=False)
    return (yield from trigger_and_read(list(detectors) + [motor]))

def one_nd_step_no_wait(detectors, step, pos_cache):
    """
    Inner loop of an N-dimensional step scan that does not wait for motors to finish moving.

    This is the same as the default function for ``per_step`` param in ND plans, except the wait is removed.

    Parameters
    ----------
    detectors : iterable
        devices to read
    step : dict
        mapping motors to positions in this step
    pos_cache : dict
        mapping motors to their last-set positions
    """
    def move():
        yield from checkpoint()
        grp = short_uid('set')
        for motor, pos in step.items():
            if pos == pos_cache[motor]:
                # This step does not move this motor.
                continue
            yield from abs_set(motor, pos, group=grp)
            pos_cache[motor] = pos
#        yield from wait(group=grp)

    motors = step.keys()
    yield from move()
    yield from trigger_and_read(list(detectors) + list(motors))

class ScanCallback(CallbackBase):
    """Future function to take place of the exisiting scan callback which is clunky.
    """
    def _createScanTab(self):
        return
    
    def _endScan(self):
        """Common function to perform end of scan activities. """
        return
    
    def start(self, doc):
        self._start_doc = doc
        
    
    def descriptor(self, doc):
        self._describe_doc = doc
        
    
    def event(self, doc):
        self._event_doc = doc
        

    def stop(self, doc):
        self._end_doc = doc
        

class DaqScan():
    """
    Class of scans involving the DAQ. This is really just a wrapper for bluesky function with some useful list generators and some functions to post to elog. The main workhorse is bluesky.list_scan
    0D scans:
        untimedRun: Start the DAQ in a free running mode. 
        timedRun: Put the DAQ in a running state and start a timer, when the timer expires, the DAQ wil stop running. Likely depracated. 
        timedRun2: Similar to timedRun, but the time is measured in the number of events, not wall time. 
    1D scans:
        lists: Scan a motor through a list of points.
        step: Scan a motor by stepping between a starting an stopping point.
        zigzag: zig-zag (or snake) scan of a motor through a list of points.
        random: randomize the order of a list and scan the motor through a list of points. The list is randomized every scan.
        tweak: Attempt to stabilize fly scans of a motor. The DAQ will interput occasionally to try to synchronize steps in a fly scan.
        adaptive: work in progress
    2D scans:
        xy_outer: scan two motors through a 2D space. The scan will take an outer product of the lists. 
        xy_zip: scan two motors together through a list of points . The scan zips the lists.
    ND scans:
        multi_motor_outer: scan multiple motors through an outer-product of lists. 
        multi_motor_zip: scan multiple motors through a zipped list.     
    Tools:
        scan.runnum(): return the most recent run number (either active or complete). 
        scan.stopRun(): Stops the current run.
        scan.log_scan(): post a message about the current scan (most recent run number) to the elog. 
        scan.show_gmd(): display a histogram of the gmd (and xgmd) data coleected during the scan.
        scan.build_lxt_ttc_list(): see tool attribute.
        scan.build_outer_list(): returns an outer product of two lists. 
    
    Attributes:
        daq: daq object
        nevents_step: number of events to take at asingle point in the scan.
        tot_events: Total number of events to save for this acquisition. Could be split into multiple runs based on max_events_run
        record: Boolean flag to record the run
        max_events_run: maximum number of events to have for a single run. Not all scans use this. 
        profile: Boolean flag to indicate you want to track the duty cycle during the scan. 
                This will use a second device scan._profile_det to check timings at different steps.
        include_gmd: Boolean flag to indicate that BlueSky should sample the gmd and xgmd data streams and include this information in the scan table.
        scan_docstring: A description of the run that is posted to the elog at the end of the scan. 
        break_up_motion: Boolean flag which will break-up 1D scans of combined movers and yield an xy_zip scan of the individual positioners. 
        
    Example: full function definition:
        >>scan.zigzag(motor1, list1, nevents_step=8000*5, tot_events=8000*5*len(list1)*5, record=True)
                    
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
        def hack_read():
            """ This is a hack for bluesky functions that want a dictionary returned from a detector when det.read() is called."""
            return {'events':{'value':self.daq.events, 'timestamp':time.time()}}
        def hack_describe():
            """ This is a temporary hack for bluesky tables."""
 #           if self.profile:
 #               return {'events': {'source':'', 'dtype':'number','shape':[]}}
 #           else:
 #               return {}
            return {'events': {'source':'', 'dtype':'number','shape':[]}}
        self.daq = daq
        self.daq.read = hack_read
        self.daq.describe = hack_describe
        self.RE = RE
        
        #### Attributes For scans
        self.nevents_step=8200*10 
        self.tot_events=1E5
        self.record = True
        self.max_events_run=0
        self.restore=False
        self.profile = False
        self.break_up_motion = False
        self.wait_for_motor = True
        self.include_gmd = True
        self.include_fzp = False
        self.include_tptc = False
        self.freeRun = True
        self._motors=[]
        self._profile_det = DaqProfile(self.daq, name='DAQ_Profiler')
        
        #### Attributes for making better tables in the elog
        self.scan_docstring = ''
        self._doc_stop=None
        self._doc_start=None
        self._doc=None
        self._f_name=None #filename for saving scan_tab.
        self._FILEPATH=None #filename for saving scan_tab
        self._num_points_RE=0 #Number of points expected at the start of the scan. 
        self._initial_pos=[]
        self._gmd_data=OrderedDict() #dictionary for saving data from the GMD/xGMD
        self._tptc_data=OrderedDict()
        self._dets = [self.daq]
        self._flyers = [] #list of devices to use as flyers
        self._per_step=None #callback function for each step of the scan.
        self._cb=None #callback for bluesky
        self._md=None #metadata for scans. 
        self._fly=None #flag to indicate fly scans.
        self._elog_gmd=True #flag to indicate posting gmd to the elog. 
        self._elog_gmd_scale=[0,0.5]
        self._elog_xgmd_scale=[0,0.5]
        
        #### Some sim motors for testing.
        self._sim_fast = FastMotor(name='sim_fast_setpoint')
        self._sim_slow = SlowMotor(name='sim_slow_setpoint')
        self._no_motor = FastMotor(name='No Motor')
        
        #stop the default table setting. 
        for i in range(0,10):
            RE.unsubscribe(i)
    
    @property
    def info(self):
        d={'Attribute': ['nevents_step', 'tot_events', 'record', 'profile', 'restore', 'include_gmd','wait_for_motor', 'max_events_run', 'freeRun', 'break_up_motion', '_elog_gmd'],
           'Values': []}
        for att in d['Attribute']:
            d['Values'].append(getattr(self, att))
        d['Attribute'].extend(['daqState','last run','RunEngine state'])
        d['Values'].extend([self.daq.daqState, self.run_num, RE.state])
        infoDF = pd.DataFrame(data=d)
        infoDF_str = tabulate(infoDF, headers="keys", tablefmt="pretty")
        print(infoDF_str)
    
    #The following are for interacting with the run engine
    #This wrapper seems less than idea.
    def resume(self):
        """Resume scan when paused."""
        if self.record: self.log_scan('Scan paused and resumed.')
        return RE.resume()    
    def abort(self):
        """Perform cleanup, then kill scan. Mark exit_stats='aborted'."""
        if self.record: self.log_scan('Scan aborted')
        return RE.abort()    
    def stop(self):
        """Perform cleanup, then kill scan. Mark exit_status='success'."""
        if self.record: self.log_scan('Scan stopped')
        return RE.stop()
    def halt(self):
        """Emergency Stop: Do not perform cleanup --- just stop."""
        if self.record: self.log_scan('Scan halted')
        return RE.halt()
    
    def _check_elog(self):
        ''' Check that the experiment number from the DAQ and the Elog match.'''
        elog_exp = elog.logbooks['experiment']
        daq_exp = next(filter(lambda s:isinstance(s,str) and 'tmo' in s,  list(self.daq.control.getStatus())), None)
        if daq_exp is None:
            return 0
        else:
            if elog_exp == daq_exp:
                return 1
        return -1
        
        
    def _update_scan_param(self, nevents_step=None, tot_events=None, max_events_run=None, record=None, restore=None, motor=None, freeRun=None, msg=None):
        """ Update the attributes of scan with the current parameters. """        
        logger.debug('nevents_step=%s, tot_events=%s, max_events_run=%s, record=%s' % (nevents_step, tot_events, max_events_run, record))
        
        #update parameters with new values
        self.nevents_step = nevents_step if nevents_step is not None else self.nevents_step
        self.tot_events = tot_events if tot_events is not None else self.tot_events
        self.max_events_run = max_events_run if max_events_run is not None else self.max_events_run
        self.record = record if record is not None else self.record
        self.restore = restore if restore is not None else self.restore
        self.freeRun = freeRun if freeRun is not None else self.freeRun
        self.scan_docstring = msg if msg is not None else self.scan_docstring
        #09/19/26: JPC removes option for fzp
        self.include_fzp = False #never use the fzp. 
        
        self._dets = [self.daq] #list of detectors for bluesky. The minimum needed to scan the DAQ is self.daq. 
        self._flyers = [] #list of flyers for bluesky.
        self._cb = {"start":self._startRun, "event":self._event, "stop": self._endRun} #callback functions for bluesky

        #setup bluesky metadata to use as a scan descritpion. 
        if self.scan_docstring != '':
            self._md = {'purpose': (self.scan_docstring + '\n\n')}
        else:
            self._md = None
                
        #Add detector for profiling the DAQ performance.         
        if self.profile:
            self._dets.append(self._profile_det)
        
        #Set daq to connected state and wait for it to get there
        self.daq._set_connected()
        logger.debug('daqState: \'%s\'' % self.daq.daqState)
        with self.daq.daqState_cv:
            while self.daq.daqState != 'connected':
                logging.debug('daqState \'%s\', waiting for \'%s\'...' % (self.daq.daqState, 'connected'))
                self.daq.daqState_cv.wait(0.5)
            logging.debug('daqState: \'%s\'' % self.daq.daqState)
#        while self.daq.daqState != 'connected':
#            continue
#        logger.debug('daqState: \'%s\'' % self.daq.daqState)
        
        #Build motor list, initial position list, and condifure DAQ. 
        if motor is None: #No motor is provided. This means the scan will be a timedRun or there was a mistake. 
            self._motors=[]
            self.daq.configure(record=self.record, events=self.nevents_step)#, motors=[self._no_motor]) #try adding later
            self._cb['all'] = [LiveTable(self._dets),self._gmd_accum]
        else: #(motor is not None)
            if isinstance(motor, list): # a list of motors was passed
                self._motors = motor            
                self._initial_pos = []
                for m in motor:
                    self._initial_pos.append(m.position)
            else: # a single motor was passed. 
                self._motors = [motor]
                self._initial_pos = [motor.position]
            self.daq.configure(motors=self._motors, record=self.record, events=self.nevents_step)
            att_list = [key for ele in self._motors for key in ele.describe() if 'setpoint' in key]
            att_list.extend([key for ele in self._dets for key in ele.describe()])
#            att_list.remove('events')
            self._cb['all'] = [LiveTable(att_list),self._gmd_accum]
#            self._cb['all'] = [LiveTable(self._motors + self._dets)]
        
        # Include readout of the gmd and xgmd in the bluesky metadata.
        if self.include_gmd:
#            self._dets.append(gmd)
#            self._dets.append(xgmd)
            self._flyers.append(gmd)
            self._flyers.append(xgmd)
        
        # Include readout of two-pulse-two-color in bluesky metadata
        if self.include_tptc:
            #self._dets.append(two_pulse_two_color)
            self._flyers.append(two_pulse_two_color)
        
        #Include spectrometer data
        #09/19/2026: JPC: Removed this function
#        if self.include_fzp:
#            self._dets.append(fzp_det)
#            self._cb['all'].append(LivePlot('fzp_data','fzp_axis'))
            
        #This can be used to ignore the status of a motor during a scan.  
        if not self.wait_for_motor:
            self._per_step = one_1d_step_no_wait
        else:
            self._per_step = None     
            
        if self.record:
            # check if the elog and daq experiment match. 
            elog_stat = self._check_elog()
            if elog_stat == 0:
                logger.warning('Could not confrim elog experiment matches daq experiment')
            elif elog_stat < 0:
                logger.error('DAQ experiment does not match elog. Restart hutch-python')
                raise Exception('Elog mismatch')
            else: #elog_stat > 0
                logger.debug('DAQ experiment matches elog experiment')
        else:
            logger.warning('Record State is set to False')

        
    def build_lxt_ttc_list(self, lxt_list:list[float], txt_window:float, add_list=None, txt_invert=False):
        """
        Build a list to scan lxt and adjust txt to keep an ATM signal in a specific window (txt_window).
        This will build lists with minimal txt movement. 
        An additional list (add_list) can be provided, and the outer product of the add_list with zip(lxt_list, txt_list) is returned.
        Example:
        >> lxt_list, txt_list, add_list = scan.build_lxt_ttc_list(*args)
        >> scan.multi_motor_zip(lxt, lxt_list, txt, txt_list, motor, add_list)
        Parameters
        ----------
        lxt_list : list[float]
            List of points to scan for lxt values.
        txt_window : float
            ATM window size.
        add_list : list[float], optional
            An additional list for creating an outer product. The default is None.
        txt_invert : bool, optional
            Direction between lxt and txt scales. Default behavior is to move the same direction.

        Returns
        -------
        lxt_list : list
            Sorted list of lxt positions
        txt_list : list
            Sorted list of txt positions for txt motor that keeps lxt within the ATM window.
        add_list : list
            Additional list as an outer product 

        """
        if isinstance(lxt_list, np.ndarray):
            lxt_list = lxt_list.tolist()
        if txt_invert: txt_dir=-1
        else: txt_dir=1
        lxt_list.sort()
        txt_list=[]
        txt_pt=txt_dir * lxt_list[0]
        for pt in lxt_list:
            if ( abs(pt - txt_dir*txt_pt) > txt_window ):
                txt_pt = txt_dir * pt
            txt_list.append(txt_pt)
        if add_list is not None:
            if isinstance(add_list, np.ndarray):
                add_list = add_list.tolist()
            tot=len(add_list)*len(lxt_list)
            add_list = _BuildList(add_list, 'z', 1, tot, 0, repeat=len(lxt_list))
            lxt_list = _BuildList(lxt_list, 'f', 1, tot, 0, repeat=1)
            txt_list = _BuildList(txt_list, 'f', 1, tot, 0, repeat=1)
            return lxt_list, txt_list, add_list
        else:
            return lxt_list, txt_list
    
    def build_outer_list(self, list1:list[float], list2:list[float], scan_type='f'):
        """
        Build an outer product of two lists to use for scans. 
        Example:
        >> list1, list2 = scan.build_outer_list(list1, list2)
        >> scan.multi_motor_zip(motor1, list1, motor2, list2, motor3, list3)
        build_outer_list creates the outer product of two lists and these are zipped with the third list for a scan.

        Parameters
        ----------
        list1 : list[float]
            List to scan fast (the inner list)
        list2 : list[float]
            List to scan slowly (the outer list)
        scan_type : TYPE, optional
            List combination type, can be (r)andom, (z)ig-zag, or (f)old. The default is 'f'.

        Returns
        -------
        List1 : list[float]
            Values from list1 after the outer product.
        List2 : list[float]
            Values from list2 after the outer product.

        """
        
        if isinstance(list1, np.ndarray):
            list1 = list1.tolist()
        if isinstance(list2, np.ndarray):
            list2 = list2.tolist()
        tot_pts = len(list1)*len(list2)
        List2 = _BuildList(list2, scan_type, 1, tot_pts, 0, repeat=len(list1)) #slow list
        List1 = _BuildList(list1, scan_type, 1, tot_pts, 0, repeat=1) #fast list
        
        return List1, List2
        
    def log_scan(self,msg=None):
        """
        Post a message about the current scan to the elog. 

        Parameters
        ----------
        msg : str
            Message to post to elog. If none is passed the function uses the string stored in scan.scan_docstring

        Returns
        -------
        None.

        """
        if msg is not None:
            elog.post(msg,run=self.run_num, tags='daq_scan')
        else:
            elog.post(self.scan_docstring,run=self.run_num, tags='daq_scan  script-generated')   

    @property    
    def run_num(self):
        """ Get the current DAQ run number"""
        runs = [item for item in daq.control.getStatus() if (isinstance(item, int) and not isinstance(item, bool))]
        if len(runs) == 0:
            run_num=None
        else:
            run_num = np.max(runs)
        return run_num
    
    def _createScanTab(self, filepath='/sdf/home/t/tmoopr/daq_scan_preproc/'):
        """ Save the scan table to an h5 file. """      
        exp_name = next(filter(lambda s:isinstance(s,str) and 'tmo' in s,  list(daq.control.getStatus())), None)
        run_num = self.run_num
        status = ((exp_name is None) or (run_num is None))
        attempt = 0
        while status: #try to get the daq to connect and give info.
            attempt+=1
            if (exp_name is None):
                exp_name = next(filter(lambda s:isinstance(s,str) and 'tmo' in s,  list(daq.control.getStatus())), None)
            if (run_num is None):
                run_num = self.run_num
            status = ((exp_name is None) or (run_num is None)) and (attempt < 11)
        if ( (exp_name is None) or (run_num is None)):
            logger.error('Something is wrong with the DAQ communication. preproc file NOT created, scan will continue.')
            return
        run_num += 1
        if not self.daq.control.getStatus()[3]:
            logger.error('Record State of DAQ is inconsistent.')
            self.daq.control.setRecord(True)
            logger.info('DAQ set to Record state.')
        #self._FILEPATH = '/cds/home/opr/tmoopr/daq_scan_preproc/' + exp_name + '/'
        self._FILEPATH = filepath + exp_name + '/'
        self._f_name = 'run%s_scantab.h5' % run_num        
        if not os.path.exists(self._FILEPATH):
            os.makedirs(self._FILEPATH)
            logger.info('Created preproc filepath: %s.' % self._FILEPATH)
        
        with h5py.File(self._FILEPATH + self._f_name, 'w') as hf_tab:
            hf_tab['names'] = ['step_value'] + [motor.name for motor in self._motors]
            hf_tab['tab'] = self._tab
            hf_tab['run_start'] = list(datetime.fromtimestamp(self._prev_evt).timetuple()[:6])
            logger.info('Saved scan table to %s' % (self._FILEPATH + self._f_name) )
            
    
    def _startRun(self, name, doc):
        self._doc_start = doc
#        logger.debug(doc)
        num_points = doc.get('num_points', 0)
        
        #setup scan table for pre-proc
        self._tab = np.zeros((num_points, len(self._motors)+1))
        self._tab[:,0] = np.arange(num_points)+1
        
        #for elog posts and scan table (h5)
        msg = doc.get('purpose', '')
        if doc['plan_name'] == 'list_scan':
            for i, motor in enumerate(doc['motors']):
                lst = doc['plan_args']['args'][2*i+1]
                self._tab[:,i+1] = lst # for pre-proc scan table
                msg=msg+'Scan %s: %s \n' % (motor, np.sort(np.unique(lst)))  
            msg=msg+'%s events/step \n' % (self.nevents_step)
        
        elif doc['plan_name'] == 'scan':
            lst = np.linspace(doc['plan_args']['args'][1], doc['plan_args']['args'][2], num_points)
            self._tab[:,1] = lst
            msg=msg+'Scan %s from %s to %s in %s steps with %s events/step.' % (doc['motors'][0], doc['plan_args']['args'][1], doc['plan_args']['args'][2], num_points, self.nevents_step)
        
        elif doc['plan_name'] == 'count':
            msg=msg+'Timed run with %s events' % self.tot_events
            
        elif doc['plan_name'] == 'tweak':
            msg=msg+'Fly scan of %s with %s events/step.\n' %(doc['motors'][0], self.nevents_step)
            self._fly=doc['motors'][0]
        
        self.scan_docstring=msg
        self._num_points_RE=num_points
                    
        # for checking data rate
        self._prev_evt=doc['time']
        if self.profile:
            self._duty_cycle=[]
        else:
            self._duty_cycle=None
        
        if self.include_gmd:
            self._gmd_data['gmd_avg_int']=[]
            self._gmd_data['xgmd_avg_int']=[]
            self._gmd_data['gmd_mj']=[]
            self._gmd_data['xgmd_mj']=[]
            if self.include_tptc:
                self._gmd_data['two_pulse_two_color_p1']=[]
                self._gmd_data['two_pulse_two_color_p2']=[]

        if self.include_fzp:
            self._gmd_data['fzp_data']=np.zeros(2048)

        if self.record:
            if self.daq.record:
                self._createScanTab()
            else:
                logger.error('DAQ failed to set state to recording')
                self.record=False                

    def _event(self, name, doc):
        self._doc = doc
        seq_num = doc.get('seq_num', None)
        data=doc.get('data', None)
        if data is not None:
            if self.profile:
                duty_cycle = float(data.get('daq_record',0) / (doc['time']-self._prev_evt) )
                self._duty_cycle.append(duty_cycle)        
            if self.include_gmd:
#                self._gmd_data['gmd_avg_int'].extend(data.get('gmd_avg_int',[]))
#                self._gmd_data['xgmd_avg_int'].extend(data.get('xgmd_avg_int',[])) 
#                self._gmd_data['gmd_mj'].extend(data.get('gmd_mj',[]))
#                self._gmd_data['xgmd_mj'].extend(data.get('xgmd_mj',[]))
                if self.include_tptc:
                    self._gmd_data['two_pulse_two_color_p1'].extend(data.get('two_pulse_two_color_p1'))
                    self._gmd_data['two_pulse_two_color_p2'].extend(data.get('two_pulse_two_color_p2'))
#            if self.include_fzp:
#                self._gmd_data.update({'fzp_data':self._gmd_data.get('fzp_data') + data.get('fzp_data')})
            if self._fly is not None:
                self._tab = np.append(self._tab, [[seq_num, data.get(self._fly,[])]], axis=0)
        self._prev_evt = doc['time']

    def _gmd_accum(self, name, doc):
        if name != 'event_page':
            return
        self._doc_page = doc
        data = doc.get('data', None)
        if data is not None:
            if 'gmd_avg_int' in data: self._gmd_data['gmd_avg_int']=data.get('gmd_avg_int',[[]])[0]
            if 'xgmd_avg_int' in data: self._gmd_data['xgmd_avg_int']=data.get('xgmd_avg_int',[[]])[0]
            if 'gmd_mj' in data: self._gmd_data['gmd_mj']=data.get('gmd_mj',[[]])[0]
            if 'xgmd_mj' in data: self._gmd_data['xgmd_mj']=data.get('xgmd_mj',[[]])[0]
            if 'two_pulse_two_color_p1' in data: self._gmd_data['two_pulse_two_color_p1']=data.get('two_pulse_two_color_p1',[[]])[0]
            if 'two_pulse_two_color_p2' in data: self._gmd_data['two_pulse_two_color_p2']=data.get('two_pulse_two_color_p2',[[]])[0]
    
    def _endRun(self, name, doc):
        self._doc_stop = doc
        num_events = doc['num_events'].get('primary',0) 
        self.scan_docstring = self.scan_docstring+('\nScan exited after %s steps with status: %s' % (num_events, doc['exit_status']) )  
        if (self._num_points_RE > 0):
            if ((doc['exit_status']=='success') and (num_events != self._num_points_RE) ):
                self.scan_docstring = self.scan_docstring+('\nWhich probably means the scan was "stopped" before it finished, not "aborted", because the scan should have had %s steps' % self._num_points_RE)
        if self.profile:
            self._duty_cycle = np.mean(self._duty_cycle[1:])
            self.scan_docstring = self.scan_docstring + ('\nAverage Duty Cycle: %2.3f' % self._duty_cycle)
        if self._fly is not None:
            self.scan_docstring = self.scan_docstring + ('\nscan points: %s' % self._tab[:,1])
        print(self.scan_docstring)
#            print('Average Duty Cycle: %s' % self._duty_cycle)
        if self.record:    
            self.log_scan()
            if ( self.include_gmd or (self._fly is not None) ):
                with h5py.File(self._FILEPATH + self._f_name, 'a') as hf_tab:
                    hf_tab['run_stop'] = list(datetime.fromtimestamp(doc['time']).timetuple()[:6])
                    if self.include_gmd:
                        for key in self._gmd_data.keys():
                            hf_tab[key] = self._gmd_data[key]
                    if self._fly is not None:
                        hf_tab['tab'] = self._tab
                        self._fly = None
                        logger.info('Saved scan tab.')
                logger.info('Saved gmd data to %s' % (self._FILEPATH + self._f_name) )
          
        os.system('echo -e "\a"')
        self.scan_docstring='' #reset docstring for the next scan. 
        if self.restore:
            for i,motor in enumerate(self._motors):
                logger.debug('moving %s to %s' % (motor.name, self._initial_pos[i]))
                motor.set(self._initial_pos[i], wait=True)
                if hasattr(motor, 'state'):
                    debug_msg = '%s.state is %s' % (motor.name, motor.state)
                else:
                    debug_msg = '%s has no attribute state' % (motor.name)
                logger.debug(debug_msg)
        
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
        self._update_scan_param(None, None, None, record, None, None)
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
        nevents = self.nevents_step
        self._update_scan_param(nevents_step=-1, tot_events=None, max_events_run=None, record=record, restore=None, motor=None)
        #self._setup_freeRun(record=self.record)       
        self.daq.control.setRecord(record)
        self.daq.control.setState('running')
        #self.daq.stage()
        #self.daq.trigger()
        self.nevents_step = nevents
        #self.daq.push_socket.send_string('running')
        #self.daq.ready.wait()
        #self.daq.ready.clear()

        with self.daq.daqState_cv:
            while self.daq.daqState != 'running':
                self.daq.daqState_cv.wait(1.0)
        print('scan.stopRun() to stop run')
        
    def _freeRun(self):
        """Free Run the DAQ after a scan."""
        #prev_rec = self.record 
        logger.info('Run complete, starting Free Run')
        #self.untimedRun(False)
        #self.record = prev_rec 
        self.daq.configure(record=False, events=-1)
        self.daq.control.setRecord(False)
        self.daq.control.setState('running')
        with self.daq.daqState_cv:
            while self.daq.daqState != 'running':
                self.daq.daqState_cv.wait(1.0)
        print('scan.stopRun() to stop run')
        
        
    def stopRun(self):
        """
        This function stops the current run and sets the record state.
        Parameters:
        ---------- 
        record: bool, 
            Flag to tell DAQ to save the run. 
        """
        self.daq._set_connected()
       
    def _endScan(self):
        """Common function to perform end of scan activities. """
        if self.include_gmd: self.show_gmd()
        if self.freeRun: self._freeRun()
        
    def timedRun2(self, tot_events=None, nevents_cycle=None, num_runs=1, record=None, freeRun=None, msg = None, motor=None):		
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
    
        msg: str,
            String providing a description of the run to append to the elog post
            
        motor: optional input of a motor name to append to the data
        """           
        
        self._update_scan_param(nevents_step=nevents_cycle, tot_events=tot_events, max_events_run=None, record=record, restore=False, motor=motor, freeRun=freeRun, msg=msg)
            
        if ( self.nevents_step > self.tot_events):
            logger.error("Error: Requested more events/cycle than total events")
            return
        num_cycles = np.ceil(self.tot_events/self.nevents_step).astype(int);
        #The run engine should not go inside the for loop. Instead the loop should be executed as below then the RE called once
        def temp_plan(num_runs, num_cycles):
            for irep in range(num_runs):
                yield from bp.count(self._dets, num=num_cycles, md=self._md)
        RE( bpp.fly_during_wrapper( temp_plan(num_runs, num_cycles), self._flyers), self._cb )
        self._endScan()
    
    def step(self, motor, start, stop, nstep, nevents_step=None, record=None, restore=True, msg=None):
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
        
        msg: str,
            String providing a description of the run to append to the elog post
             
        """

        self._update_scan_param(nevents_step=nevents_step, tot_events=None, max_events_run=None, record=record, restore=restore, motor=motor, msg=msg)       
        RE( bpp.fly_during_wrapper( bp.scan(self._dets, motor, start, stop, nstep, per_step=self._per_step, md=self._md), self._flyers), self._cb)
        self._endScan()                         
           
    def _lists(self, motor, posList):
        """
        Scan "motor" through a list of points. 
        """    
#        self._update_scan_param(nevents_step=nevents_step, tot_events=None, max_events_run=None, record=record, restore=restore, motor=motor, msg=msg)

        if isinstance(posList, np.ndarray):
            posList = posList.tolist()
        if not isinstance(posList, list):
            print('you need to pass a list of steps')
            return      
        #Scan the list
        if ( any( isinstance(el, list) for el in posList ) ): #posList is list of lists. Need multiple runs
            def temp_plan(posList): #define plan to iterate over runs
                for lst in posList:
                    #yield from bp.list_scan(self._dets, motor, lst, per_step=self._per_step)
                    yield from bp.list_scan(self._dets, motor, list(map(float, lst)), per_step=self._per_step, md=self._md)
            RE( bpp.fly_during_wrapper( temp_plan(posList), self._flyers), self._cb )
                
        else: #posList is a list of points to visit  
            #RE(bp.list_scan(self._dets, motor, posList, per_step=self._per_step), self._cb)
            RE( bpp.fly_during_wrapper( bp.list_scan(self._dets, motor, list(map(float, posList)), per_step=self._per_step, md=self._md), self._flyers), self._cb)
        
        self._endScan()
            
    def lists(self, motor, posList, nevents_step=None, record=None, restore=True, msg=None):
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
            
        msg: str,
           String providing a description of the run to append to the elog post
        
        """
    
        self._update_scan_param(nevents_step=nevents_step, tot_events=None, max_events_run=None, record=record, restore=restore, motor=motor, msg=msg)
        self._lists(motor, posList) 
        
    def random(self, motor, posList, nevents_step=None, tot_events=None, record=None, restore=True, max_events_run=None, msg=None):
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

        msg: str,
           String providing a description of the run to append to the elog post
        
        """
#        self._update_scan_param(nevents_step, tot_events, max_events_run, record, restore, motor)
        self._update_scan_param(nevents_step=nevents_step, tot_events=tot_events, max_events_run=max_events_run, record=record, restore=restore, motor=motor, msg=msg)
        if self.tot_events < 60:
            logger.warning('Scan requested less than 60 total events, assuming this was the total number of scans.')
            self.tot_events = self.tot_events * self.nevents_step * len(posList)
        #Build List
        fullList = _BuildList(posList, 'r', self.nevents_step, self.tot_events, self.max_events_run, repeat=1 )
        if fullList is None:
            return

        #use list_scan to scan this list. 
#        self._lists(motor, fullList, self.nevents_step, self.record, restore) 
        self._lists(motor, fullList)
    
    def zigzag(self, motor, posList, nevents_step=None, tot_events=None, record=None, restore=True, max_events_run=None, msg=None):
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
            
        msg: str,
            String providing a description of the run to append to the elog post
         
        """
        if isinstance(motor, SyncAxis) and self.break_up_motion:
            logger.warning('%s is a combined mover. Breaking motion into axes' % motor.name)
            argList = []
            for positioner in motor._real_attrs:
                argList.append(getattr(motor, positioner))
                newList = motor.scales[positioner] * np.array(posList) + motor.offsets[positioner]
                argList.append(newList)
            self.xy_zip(*tuple(argList), nevents_step=nevents_step, tot_events=tot_events, record=record, restore=restore, max_events_run=max_events_run, msg=msg)
            return
#        self._update_scan_param(nevents_step, tot_events, max_events_run, record, restore, motor)
        self._update_scan_param(nevents_step=nevents_step, tot_events=tot_events, max_events_run=max_events_run, record=record, restore=restore, motor=motor, msg=msg)
        if self.tot_events < 60:
            logger.warning('Scan requested less than 60 total events, assuming this was the total number of scans.')
            self.tot_events = self.tot_events * self.nevents_step * len(posList)           
        #Build List
        fullList = _BuildList(posList, 'z', self.nevents_step, self.tot_events, self.max_events_run, repeat=1 )
        if fullList is None:
            return                  

        #use list_scan to scan this list. 
        self._lists(motor, fullList) 
            
    def _lists_dual(self, x_motor, x_list, y_motor, y_list):
        """
        Scan two motors (x and y) through a list of points (x_list and y_list). 
                 
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

#        self.daq.control.setState('connected')
#        self._update_scan_param(nevents_step, None, None, record, restore, [x_motor, y_motor])                                
#        self._update_scan_param(nevents_step=nevents_step, tot_events=None, max_events_run=None, record=record, restore=restore, motor=[x_motor, y_motor], msg=msg)
    
        def temp_plan(x_List, y_List): #define plan to iterate over runs
            for i, (x_lst, y_lst) in enumerate( zip( x_List, y_List) ):
                #yield from bp.list_scan(self._dets, x_motor, x_lst, y_motor, y_lst, per_step=self._per_step)
                yield from bp.list_scan(self._dets, x_motor, list(map(float, x_lst)), y_motor, list(map(float, y_lst)), per_step=self._per_step, md=self._md)

        RE( bpp.fly_during_wrapper( temp_plan(x_list, y_list), self._flyers), self._cb )  
        
        self._endScan()           
    
    def lists_dual(self, x_motor, x_list, y_motor, y_list, nevents_step=None, record=None, restore=True, msg=None):
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
            
        msg: str,
            String providing a description of the run to append to the elog post
         
        """
        self._update_scan_param(nevents_step=nevents_step, tot_events=None, max_events_run=None, record=record, restore=restore, motor=[x_motor, y_motor], msg=msg)
        self._lists_dual(x_motor, x_list, y_motor, y_list)
        
    def xy_outer(self, x_motor, x_list, y_motor, y_list, nevents_step=None, tot_events=None, record=None, scan_type =['z','z'], restore=True, max_events_run=None, msg=None):
        """
        Scan two motors (x and y). This function will scan the "x_motor" through the "x_list" while also scanning through the "y_list" with the "y_motor," taking "nevents_step" per step until it has completed at least "tot_events".
        The 'scan_type' input specifies wether the motor is scanned (r)andomly or (z)ig-zagged. The default is to zig-zag scan both motors.  
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
            
        msg: str,
            String providing a description of the run to append to the elog post
         
        """
#        self._update_scan_param(nevents_step, tot_events, max_events_run, record, restore,[x_motor,y_motor])
        self._update_scan_param(nevents_step=nevents_step, tot_events=tot_events, max_events_run=max_events_run, record=record, restore=restore, motor=[x_motor, y_motor], msg=msg)        
        if self.tot_events < 60:
            logger.warning('Scan requested less than 60 total events, assuming this was the total number of scans.')
            self.tot_events = self.tot_events * self.nevents_step * len(x_list) * len(y_list)
        #Build Lists
        y_List = _BuildList(y_list, scan_type[1], self.nevents_step, self.tot_events, self.max_events_run, repeat=len(x_list))
        if not isinstance(y_List[0], list):
            y_List = [y_List]
        x_List = _BuildList(x_list, scan_type[0], self.nevents_step, sum([len(ele) for ele in y_List])*self.nevents_step, self.max_events_run, repeat=1)
        if not isinstance(x_List[0], list):
            x_List = [x_List]
        if ( (y_List[0] is None) or (x_List[0] is None) ):
            return

        self._lists_dual(x_motor, x_List, y_motor, y_List)
    
    def xy_zip(self, x_motor, x_list, y_motor, y_list, nevents_step=None, tot_events=None, record=None, restore=True, max_events_run=None, msg=None):
        """
        Scan two motors (x and y). This function will scan the "x_motor" through the "x_list" while also scanning through the "y_list" with the "y_motor," taking "nevents_step" per step until it has completed at least "tot_events". "x_list" and "y_list" are sorted according to "x_list".
        
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
            
        msg: str,
            String providing a description of the run to append to the elog post
         
        """
#        self._update_scan_param(nevents_step, tot_events, max_events_run, record, restore, [x_motor,y_motor])
        self._update_scan_param(nevents_step=nevents_step, tot_events=tot_events, max_events_run=max_events_run, record=record, restore=restore, motor=[x_motor, y_motor], msg=msg)   
        if self.tot_events < 60:
            logger.warning('Scan requested less than 60 total events, assuming this was the total number of scans.')
            self.tot_events = self.tot_events * self.nevents_step * len(x_list)
        #Build Lists
        if len(x_list) != len(y_list):
            print('x_list and y_list must have the same length.')
            return
        order = np.argsort(x_list)
        x_list = np.array(x_list)[order].tolist()
        y_list = np.array(y_list)[order].tolist()
        x_List = _BuildList(x_list, 'z', self.nevents_step, self.tot_events, self.max_events_run, repeat=1 )
        if not isinstance(x_List[0], list):
            x_List = [x_List] 
        y_List = _BuildList(y_list, 'f', self.nevents_step, self.tot_events, self.max_events_run, repeat=1 )
        if not isinstance(y_List[0], list):
            y_List = [y_List]
        if ( (y_List[0] is None) or (x_List[0] is None) ):
            return
        
        self._lists_dual(x_motor, x_List, y_motor, y_List)  

           
    def multi_motor_outer(self, *args, nevents_step=None, tot_events=None, record=None, restore=True, msg=None):
        """
        Scan multiple motors through an outer product of lists. The first scan speed of the first motor is fastest.
        
        Parameters:
        -----------
        *args: 
            input of motor and list:
                motor1, [point1, point2, ...], #fastest scan
                motor2, [point1, point2, ...],
                ...,
                motorN, [point1, point2, ...] #slowest scan
    
        nevents_step: int, 
            Number of events to take at each position. 
    
        tot_events: int, 
            Total number of events to take
    
        record: bool, 
            Flag to tell DAQ to save the run. 
    
        restore: bool, 
            Returns the motor to the starting position after the scan
            
        msg: str,
            String providing a description of the run to append to the elog post
        """

        motors=[]
        input_lists = []
        #initial_pos=[]
        argList=[]
        Tot_events = self.nevents_step
        for i, (motor, pos_list) in enumerate(partition(2, args)): 
            motors.append(motor)
        #    initial_pos.append(motor.position)
            if isinstance(pos_list, np.ndarray):
                pos_list = pos_list.tolist()
            pos_list.sort()
            input_lists.append(pos_list)
            Tot_events = Tot_events * len(pos_list)
            argList.append(motor)
            argList.append([])
        
#        self._update_scan_param(nevents_step, tot_events, None, record,restore, motors)
        self._update_scan_param(nevents_step=nevents_step, tot_events=tot_events, max_events_run=None, record=record, restore=restore, motor=motors, msg=msg)             
        if self.tot_events < 60:
            logger.warning('Scan requested less than 60 total events, assuming this was the total number of scans.')
            self.tot_events = Tot_events * self.tot_events
        else:
            self.tot_events = np.ceil(tot_events/Tot_events) * Tot_events
        repeat=1
        for (i,List) in enumerate(input_lists):
            #argList[2*i+1] = _BuildList(List, 'z', self.nevents_step, self.tot_events, max_events_run=-1, repeat=repeat)
            argList[2*i+1] = _BuildList(list(map(float, List)), 'z', self.nevents_step, self.tot_events, max_events_run=-1, repeat=repeat)
            repeat = repeat * len(List)
                    
        RE( bpp.fly_during_wrapper( bp.list_scan(self._dets, *tuple(argList), per_step=self._per_step, md=self._md), self._flyers), self._cb )
        
        self._endScan()          

    def multi_motor_zip(self, *args, nevents_step=None, tot_events=None, record=None, restore=True, msg=None):
        """
        Scan multiple motors through a zipped list.
        
        Parameters:
        -----------
        *args: 
            input of motor and list:
                motor1, [point1, point2, ...], 
                motor2, [point1, point2, ...],
                ...,
                motorN, [point1, point2, ...] 
    
        nevents_step: int, 
            Number of events to take at each position. 
    
        tot_events: int, 
            Total number of events to take
    
        record: bool, 
            Flag to tell DAQ to save the run. 
    
        restore: bool, 
            Returns the motor to the starting position after the scan

        msg: str,
            String providing a description of the run to append to the elog post
        """

        motors=[]
        input_lists = []
#        initial_pos=[]
        argList=[]
        for i, (motor, pos_list) in enumerate(partition(2, args)): 
            motors.append(motor)
#            initial_pos.append(motor.position)
            if isinstance(pos_list, np.ndarray):
                pos_list = pos_list.tolist()
            if i == 0:
                order = np.argsort(pos_list)
            input_lists.append(np.array(pos_list)[order].tolist())
            argList.append(motor)
            argList.append([])
        
#        self._update_scan_param(nevents_step, tot_events, None, record,restore, motors)
        self._update_scan_param(nevents_step=nevents_step, tot_events=tot_events, max_events_run=None, record=record, restore=restore, motor=motors, msg=msg)         
        tot_events = self.nevents_step * len(pos_list)
        if self.tot_events < 60:
            logger.warning('Scan requested less than 60 total events, assuming this was the total number of scans.')
            self.tot_events = tot_events * self.tot_events
        else:
            self.tot_events = np.ceil(self.tot_events/tot_events) * tot_events
                
        for (i,List) in enumerate(input_lists):
            #argList[2*i+1] = _BuildList(List, 'f', self.nevents_step, self.tot_events, max_events_run=-1, repeat=1)
            argList[2*i+1] = _BuildList(list(map(float,List)), 'f', self.nevents_step, self.tot_events, max_events_run=-1, repeat=1)
            
        RE( bpp.fly_during_wrapper( bp.list_scan(self._dets, *tuple(argList), per_step=self._per_step, md=self._md), self._flyers), self._cb ) 
        
        self._endScan()
                       
    def tweak(self, motor, step=0, nevents_step=None, record=None, restore=False, msg=None):
        self.profile=False
        self._update_scan_param(nevents_step=nevents_step, tot_events=None, max_events_run=None, record=record, restore=restore, motor=motor, msg=msg)                 
        RE( bpp.fly_during_wrapper( bp.tweak(self.daq,'events', motor, step, md=self._md), self._flyers), self._cb )
        self._endScan()
    
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
        self._update_scan_param(nevents_cycle, None, None, record, False, motor)       
        #Setup DAQ
        self.daq.control.setState('connected')
        self._motors=[motor]
        self._initial_pos=[]
        self.daq.configure(record=record, events=self.nevents_step, motors=self._motors)
        dets = [self.daq, feedback_device]
    
    #def _adaptive_step_find_scan(detectors, motor, start, limits, thresholds, num=1):

    def show_gmd(self):
        """ Plots the GMD data from the last run."""
        if not bool(len(self._gmd_data.keys())):
            logger.error('No GMD Data to display!')
            return
        else:
            fig1 = plt.figure(101)
            tptc_bin = np.arange(0,.05,0.0001)
            gmd_bin = np.arange(0,0.601, 0.001)
            
            if self.include_tptc:
                plt.subplot(1,2,1)
                plt.hist(self._gmd_data['two_pulse_two_color_p1'],tptc_bin)
                plt.title('P1 pulse energy')
                plt.xlabel('pulse energy (mJ)')
                if self._elog_gmd_scale is not None:
                    plt.xlim(self._elog_gmd_scale)
                else:
                    plt.xlim([0,0.005])
                plt.subplot(1,2,2)
                plt.hist(self._gmd_data['two_pulse_two_color_p2'],tptc_bin)
                plt.title('P2 pulse energy')
                plt.xlabel('pulse energy (mJ)')
                if self._elog_xgmd_scale is not None:
                    plt.xlim(self._elog_xgmd_scale)
                else:
                    plt.xlim([0,0.005])
            else:
                plt.subplot(1,2,1)
                plt.hist(self._gmd_data['gmd_mj'],gmd_bin)
                plt.title('gmd pulse-to-pulse')
                plt.xlabel('pulse energy (mj)')
                if self._elog_gmd_scale is not None:
                    plt.xlim(self._elog_gmd_scale)
                else:
                    plt.xlim([0,0.5])
                plt.subplot(1,2,2)
                plt.hist(self._gmd_data['xgmd_mj'],gmd_bin)
                plt.title('xgmd pulse-to-pulse')
                plt.xlabel('pulse energy (mj)')
                if self._elog_xgmd_scale is not None:
                    plt.xlim(self._elog_xgmd_scale)
                else:
                    plt.xlim([0,0.1])
            if self.include_fzp:
                fig2 = plt.figure(102)
                plt.plot(self._gmd_data['fzp_data'])
                plt.title('Average Spectrum from SP1K4-GIGE')
                plt.xlabel('pixel')
        
            if ( self.record and self._elog_gmd ):
                fig1.savefig('tmoopr_elog_plots/gmd_fig.png')
                if self.include_fzp:
                    fig2.savefig('tmoopr_elog_plots/fzp_fig.png')
                run_num = self.run_num
                if run_num is not None:
                    elog.post(('Pulse energy data for run %s' % (run_num)),attachments=['tmoopr_elog_plots/gmd_fig.png'], run=run_num, tags='daq_scan script-generated')
                    if self.include_fzp:
                        elog.post(('Avg FZP Spectrum for run %s' % (run_num)),attachments=['tmoopr_elog_plots/fzp_fig.png'], run=run_num, tags='daq_scan script-generated')
                else:
                    elog.post('Pulse energy data', attachments=['tmoopr_elog_plots/gmd_fig.png'], tags='daq_scan script-generated')
                   
    def check_supports(self, device=None):
        """ Check the protocols that a device might need to be used as a motor"""
        device = device if device is not None else self._motors[0]
        protocol_list = [Stageable, Triggerable, Readable, Locatable, Movable, Flyable, Subscribable]
        requirements_list = ["Stageable'>", "Triggerable'>", "Readable'>", "Movable'>"]
        protocol_dict = OrderedDict()
        require_dict = OrderedDict()
        has_protocol = []
        for protocol in protocol_list:
            key = repr(protocol).split('.')[-1]
            result = isinstance(device, protocol)
            if result:
                has_protocol.append(key)
            protocol_dict.update({key:result})
            require_dict.update({key:(key in requirements_list)})
        infoDF = pd.DataFrame({"Protocol:": protocol_dict.keys(), "Result:": protocol_dict.values(), 'Required:': require_dict.values()})
        infoDF_str = tabulate(infoDF, headers="keys", tablefmt="pretty")
        print(infoDF_str)
        all_met = set(requirements_list).issubset(has_protocol)
        return all_met
            
    

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

def get_hsd(py_file=None, log=False, **kwargs):
    if py_file is None:
        py_file = 'daq/scripts/tmo.py'
    with open(py_file,'r') as f:
        stop_read = False
        read_str = ''
        while not stop_read:
            line = f.readline()
            if line.startswith('from'):
                stop_read = True
            else:
                read_str+=line
    if 'hsd_dict' in read_str:
        hsd_dict = eval(read_str.split(sep='=')[1])
        host_dict = OrderedDict()
        for key,value in hsd_dict.items():
            for hsd_proc in procmgr_hsd:
                if hsd_proc['id'] == value:
                    host_dict[key] = hsd_proc['host']
              
        hsdDF = pd.DataFrame({"HSD Labels:":hsd_dict.keys(), "PCI Slot:":hsd_dict.values(), "Host:":host_dict.values()})
        hsdDF_str = tabulate(hsdDF, headers="keys", tablefmt="pretty")
        print(hsdDF_str)
        if log:
            kwargs['tags'] = 'hsd_config ' + kwargs.get('tags','')
            elog.post(hsdDF.to_string(), **kwargs) 
    else:
        logger.error('py_file does not contain hsd_dict in header.')
      
class FzpDataWoDaq(BaseInterface, Device):
    tab_component_names= True
    fzp_data = Cpt(EpicsSignalRO, 'tmo:ana:acr_graph:data:fzp', auto_monitor=False, kind='hinted')
    trigger_signals = [fzp_data]
    
    def __init__(self, **kwargs):
        curr_list = os.getenv('EPICS_CA_ADDR_LIST', '')
        new_add = '172.21.152.49:5064'
#        if new_add not in curr_list.split():
#            os.environ['EPICS_CA_ADDR_LIST'] = f"{curr_list} {new_add}".strip()
#        os.environ['EPICS_CA_AUTO_ADDR_LIST'] = 'NO'
        return super().__init__(**kwargs)
    
    def stage(self):
        """"Needed for future use"""
        return [self]
    
    def trigger(self):
        """Call that is used by bluesky prior to read()"""
        def check_value(*, old_value, value, **kwargs):
            """Mark status as finished when the number changes"""
            return not(np.array_equal(old_value, value))        
        status = SubscriptionStatus(self.fzp_data, check_value)
        return status
            
    def read(self):
        """Put the signal into a simple dictionary for bluesky to read"""
        value = self.fzp_data.get()
        self.data=OrderedDict()
        self.data['fzp_data']={'value':value, 'timestamp':self.fzp_data.timestamp}
        self.data['fzp_axis']={'value':range(2048), 'timestamp':self.fzp_data.timestamp}
        return self.data
    
    def unstage(self):
        """"Needed for future use"""
        return [self]
    
    def describe(self):
        """Return a dictionary which describes the format of the data returned from read()"""
        data = OrderedDict()
        data['fzp_data'] = {'source':self.fzp_data.pvname, 'dtype':'array','shape':[2048,1], 'units':'adu'}
        data['fzp_axis'] = {'source':'fixed', 'dtype':'array','shape':[2048,1], 'units':'adu'}
        return data
    
class FzpDataWDaq(FzpDataWoDaq):
        
    def stage(self):
        def data_cb(self, value, timestamp, **kwargs):
            self.data['fzp_data']['value'] += value
            self.data['fzp_data']['timestamp'] = timestamp
            
        self.data=OrderedDict()
        self.data['fzp_data']={'value':np.zeros(2048), 'timestamp':0}
        self.uid = self.fzp_data.subscribe(data_cb)    
        return [self]
    
    def trigger(self):
        return DeviceStatus(self, done=True, success=True)
            
    def read(self):
        return self.data
    
    def unstage(self):
        self.fzp_data.unsubscribe(self.uid)
        return [self]
    
    def describe(self):
        return {'fzp_data': {'source':self.fzp_data.pvname, 'dtype':'array','shape':[2048,1], 'units':'adu'}}

tmo_fzp_data = FzpDataWoDaq(name='tmo_fzp_data') 

        