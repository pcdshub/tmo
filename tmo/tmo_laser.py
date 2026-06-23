from pcdsdevices.device import Device
from ophyd.status import DeviceStatus, SubscriptionStatus
from pcdsdevices.interface import BaseInterface
from pcdsdevices.pv_positioner import PVPositionerIsClose
from pcdsdevices.epics_motor import SmarActTipTilt, SmarActEncodedTipTilt, EpicsMotorInterface, SmarAct, SmarActOpenLoop
from pcdsdevices.digitizers import Qadc134
from pcdsdevices.lasers.elliptec import EllRotation, Ell9
from ophyd.device import Component as Cpt
from ophyd.device import FormattedComponent as FCpt
from ophyd.signal import EpicsSignal, EpicsSignalRO, Signal
#from ophyd.pseudopos import PseudoSingle
from tmo.db import li1k4, tm1k4#, lm1k4_qadc
#from tmo.db import elog
#from tmo.tmo_sequencer import tmo_evr2
from tmo.db import RE, bp
import lmfit
from bluesky.callbacks import LiveFit
from bluesky.callbacks.mpl_plotting import LiveFitPlot, LivePlot
from tmo.tmo_utils import log_positions as _log_positions
from tmo.tmo_utils import add_tags as _add_tags
from tmo.tmo_utils import EpicsSignalROStop
from collections import OrderedDict, deque
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import datetime
import logging
logger = logging.getLogger(__name__)


class LaserMirrorMotor(PVPositionerIsClose):
    setpoint = Cpt(EpicsSignal, '')
    readback = Cpt(EpicsSignalRO, '.RBV')
    atol = 0.1
    rtol = 0

class WaveplateMotor(PVPositionerIsClose):
    setpoint = Cpt(EpicsSignal, ':MOVE')
    readback = Cpt(EpicsSignalRO, ':CURPOS')
    home_cmd = Cpt(EpicsSignal, ':HOME')
    atol = 0.1
    rtol = 0

    def home(self):
        self.home_cmd.put(1)

class LaserDelayA(PVPositionerIsClose):
    setpoint = Cpt(EpicsSignal, ':DELAY_SET')
    readback = Cpt(EpicsSignalRO, ':GET_TIC_NS')
    

class LaserDelayB(PVPositionerIsClose):
    setpoint = Cpt(EpicsSignal, '_SET')
    readback = Cpt(EpicsSignalRO, '_RBV')

lm1k4_qadc = Qadc134('LM1K4:QADC:01', name='lm1k4_qadc')

class LaserShutter(BaseInterface, Device):
    """
    Laser Shutter Object.
    
    """
    
    state = Cpt(EpicsSignal, '_SHUTTER_24V', kind='normal', doc='State of shutter')
    tab_component_names = True
    tab_whitelist = ['open', 'close']
    
    def open(self):
        """ Open the shutter."""
        self.state.put(1)
        
    def close(self):
        """ Close the shutter."""
        self.state.put(0)
        
    @property
    def position(self):
        """The current positrion of the shutter"""
        states = ['close', 'open']
        return states[self.state.get()]


class SmarPodAxis(PVPositionerIsClose):
    setpoint = FCpt(EpicsSignal, '{prefix}:CMD:{ch_name}')
    readback = FCpt(EpicsSignal, '{prefix}:{ch_name}')
    actuate = Cpt(EpicsSignal, ':CMD:MOVE')
    actuate_value = 0
    stop_signal = Cpt(EpicsSignal, ':CMD:STOP')
    stop_value = 0
    reachable = Cpt(EpicsSignal, ':CMD:REACHABLE')
    atol = 0.005

    def __init__(self, prefix, ch=None, **kwargs):
        self.ch_name = ch
        return super().__init__(prefix, **kwargs)   

    def _setup_move(self, position):
        '''Move and do not wait until motion is complete (asynchronous).
        Before moving check the position is reachable.'''
        logger.debug('%s.setpoint=%s', self.name, position)
        self.setpoint.put(position, wait=True)

        logger.debug('Checking position is reachable')
        if not self.reachable.get():
            logger.error('Position not reachable. Move aborted.')
            return

        if self.actuate is not None:
            logger.debug('%s.actuate=%s', self.name, self.actuate_value)
            self.actuate.put(self.actuate_value, wait=False)

class SmarPod(BaseInterface, Device):
    """
    Class for Smarpod combined motion
    """
    tab_component_names = True
    
    actuate = Cpt(EpicsSignal, ':CMD:MOVE') 
    stop_signal = Cpt(EpicsSignal, ':CMD:STOP')
    sync = Cpt(EpicsSignal, ':CMD:SYNC')
    
    x = Cpt(SmarPodAxis, '', ch='Y', egu ='mm', kind='normal')
    y = Cpt(SmarPodAxis, '', ch='Z', egu ='mm', kind='normal')
    z = Cpt(SmarPodAxis, '', ch='X', egu ='mm', kind='normal')
    rx = Cpt(SmarPodAxis, '', ch='RY', egu ='deg', kind='normal')
    ry = Cpt(SmarPodAxis, '', ch='RZ', egu ='deg', kind='normal')
    rz = Cpt(SmarPodAxis, '', ch='RX', egu ='deg', kind='normal')

class SmarPodMotor(BaseInterface, Device):
    """
    Class to move each motor of smarpod independently, i.e. without the kinematic correction. 
    """
    tab_component_names = True
    
    m1 = Cpt(SmarAct, ":CH1", kind='normal')
    m2 = Cpt(SmarAct, ":CH2", kind='normal')
    m4 = Cpt(SmarAct, ":CH4", kind='normal')
    m5 = Cpt(SmarAct, ":CH5", kind='normal')
    m6 = Cpt(SmarAct, ":CH6", kind='normal')

class SmarActThreeAxis(BaseInterface, Device):
    """
    This class combines three SmarAct stages into a single device. Used for lens.
    """
    tab_component_names=True
    x = Cpt(SmarAct, '_LM2', kind='normal', 
                 doc='optic motion x')
    y = Cpt(SmarAct, '_LM3', kind='normal', 
                 doc='optic motion y')
    z = Cpt(SmarAct, '_LM1', kind='normal', 
                 doc='optic motion z')
        
class SmarActDual(BaseInterface, Device):
    """
    This class controls the dual carriage smaract stages and uses motion presets to make correlated movements of of the carriages..
    """
    tab_component_names= True
    tab_whitelist = ['position','state','set_current_position','list_states']
    
    mr1 = Cpt(SmarAct, '_MR1', kind='normal', doc='first mirror on dual carriage smarAct')
    mr2 = Cpt(SmarAct, '_MR2', kind='normal', doc='second mirror on dual carriage smarAct')
        
    def state(self,state=None):
        """
        Calling the object will either move the object or get the current position, depending on the state arguement.
        preset positions can be defined with the set_current_position method, or using the individual preset class for the two motors. 
        """
        if self.mr1.presets.sync_needed(): self.mr1.presets.sync()
        if self.mr2.presets.sync_needed(): self.mr2.presets.sync()
        
        if state is None:
            mr1_state = self.mr1.presets.state()
            mr2_state = self.mr2.presets.state()
            if ( (mr1_state == 'Unknown') or (mr2_state == 'Unknown') ):
                state_out = 'Unknown'
            elif mr1_state != mr2_state:
                state_out = 'Unknown'
            else:
                state_out = mr1_state
            return state_out
        else:
            if ( hasattr(self.mr1.presets.positions, state) and hasattr(self.mr2.presets.positions, state) ):
                mv1=getattr(self.mr1, ('umv_'+state))
                mv2=getattr(self.mr2, ('umv_'+state))
                mv1()
                mv2()
            else:
                logger.error('One of the motors is missing the desired state.')
            return None
    
    @property
    def position(self):
        return self.state()
        
    def set_current_position(self, name, preset_type='exp', *kwargs):
        """
        Define a preset position with 'name' for the current position of the mirrors.
        preset_type is used to decide if this will be for the (exp)eriment or the hutch.
        you can pass an optional comment to describe the preset. 
        """
        if preset_type == 'exp':
            self.mr1.presets.add_here_exp(name, *kwargs)
            self.mr2.presets.add_here_exp(name, *kwargs)
        elif preset_type == 'hutch':
            self.mr1.presets.add_here_hutch(name, *kwargs)
            self.mr2.presets.add_here_hutch(name, *kwargs)
        else:
            logger.error('preset_type not recognizied. Should be "exp" or "hutch".')

    def list_states(self):
        """
        List the possible states for each motor. 
        """                
        if self.mr1.presets.sync_needed(): self.mr1.presets.sync()
        if self.mr2.presets.sync_needed(): self.mr2.presets.sync()
        print(self.mr1.presets.positions)
        print(self.mr2.presets.positions)


#class QadcEdgeDiff (BaseInterface, Device):
#    edge_1 = Cpt(EpicsSignalRO, '_EDGE1', kind='omitted')
#    edge_2 = Cpt(EpicsSignalRO, '_EDGE2', kind='omitted')

class LaserTimingAlarm(BaseInterface, Device):
    
    tab_component_names= True
    tab_whitelist = ['position', 'diff', 'set_alarm']#, 'setup_qadc']
    
#    qadc = Cpt(TmoQadc134, '', kind='omitted')
    edge = FCpt(EpicsSignalRO,'{prefix}:{edge_pv}',kind='hinted')
    edge1_lolo=FCpt(EpicsSignal, '{prefix}:{edge_pv}.LOLO', kind='omitted')
    edge1_low=FCpt(EpicsSignal, '{prefix}:{edge_pv}.LOW', kind='omitted')
    edge1_high=FCpt(EpicsSignal, '{prefix}:{edge_pv}.HIGH', kind='omitted')
    edge1_hihi=FCpt(EpicsSignal, '{prefix}:{edge_pv}.HIHI', kind='omitted')
    
    def __init__(self, prefix, edge_pv=None, **kwargs):
        self.edge_pv = edge_pv
        return super().__init__(prefix, **kwargs)
    
    def set_alarm(self):
        """Set the alaram based on the current values."""
        #This should probably start by calling the setup_qadc, in case it is not configured.
        curr_edge = self.edge.get()
        self.edge1_low.put(curr_edge - 0.11)
        self.edge1_lolo.put(curr_edge - 0.15)
        self.edge1_high.put(curr_edge + 0.11)
        self.edge1_hihi.put(curr_edge + 0.15)
        self.edge1_set = curr_edge
        
    # def setup_qadc(self, trig_event=258, trig_delay=14649, trace_len=64000):
    #     """Configure the qadc to monitor timing"""
    #     self.qadc.start.put(0)
    #     self.qadc.full_en.put(1)
    #     self.qadc.ichan.put(0)
    #     self.qadc.interleave.put(1)
    #     self.qadc.length.put(trace_len)
    #     self.qadc.prescale.put(1)
    #     self.qadc.trig_delay.put(trig_delay)
    #     self.qadc.trig_event.put(trig_event)
    #     self.qadc.config.put(1)
    #     self.qadc.start.put(1)
                
    @property
    def position(self):
        """Current edge value from the qadc [in ns]"""
        if self.edge.connected:
            return self.edge.get()
        else:
            return None
    
    @property
    def diff(self):
        """Differece between the alarm set value and the current position on the qadc [in ns]"""
        edge_set = getattr(self, 'edge1_set', None)
        if edge_set is None:
            diff = 0
        else:
            diff = edge_set - self.position
            
        return diff    
    
class TmoQadc134(Qadc134):
    """
    QADC interface for laser timing.
    
    Methods
    -------
    
    avg_config
        Setup the averaging parameters for the QADC (number of events to average, channel to average, and number of peaks to look for.)
    
    stage
        Configure the device to start running average
        
    get_diff
        Perform the measurement and print the results to the screen    
    
    elog
        post the results (stored in self.data) to the elog
    
    unstage
        Staging the device launches a thread to continously poll the QADC for fast sampling. This method stops that thread.
    
    save_raw_traces
        save raw qadc (out0 or out1) data to a file. The saved traces are also stored in qadc.saved_traces.
    
    Examples
    --------
    
    Make a measurement:
        >>>> qadc.avg_config(events=10, channel=1, num_peaks=2) #look for 2 peaks on channel 1 and 50 averages
        >>>> qadc.stage()
        >>>> qadc.get_diff() #return the measurement
        >>>> qadc.elog(msg='last timing measurement') #post the result to the elog
        >>>> qadc.unstage() # stop the device
    """
    
    tab_component_names=True
    tab_whitelist = ['avg_config', 'get_diff', 'save_raw_traces', 'elog']
        
    
    out0_edges = Cpt(EpicsSignalRO, ':OUT0_CFD_EDGES', kind='normal',
                     doc='list of all edges found on out 0')
    out1_edges = Cpt(EpicsSignalRO, ':OUT1_CFD_EDGES', kind='normal',
                     doc='list of all edges found on out 1')    
    out0_edge1 = Cpt(EpicsSignalRO, ':OUT0_EDGE1', kind='normal',
                     doc='first edge found on out0')
    out0_edge2 = Cpt(EpicsSignalRO, ':OUT0_EDGE2', kind='normal',
                     doc='second edge found on out0')
    out1_edge1 = Cpt(EpicsSignalRO, ':OUT1_EDGE1', kind='normal',
                     doc='first edge found on out1')
    out1_edge2 = Cpt(EpicsSignalRO, ':OUT1_EDGE2', kind='normal',
                     doc='second edge found on out1')
    #background
    out0_bg_correction_enable = Cpt(EpicsSignalRO, ':OUT0_BGCORR', kind='config')
    out0_bg_start = Cpt(EpicsSignal, ':OUT0_BGSTART', kind='config')
    out0_bg_end = Cpt(EpicsSignal, ':OUT0_BGEND', kind='config')
    out1_bg_correction_enable = Cpt(EpicsSignalRO, ':OUT1_BGCORR', kind='config')
    out1_bg_start = Cpt(EpicsSignal, ':OUT1_BGSTART', kind='config')
    out_bg_end = Cpt(EpicsSignal, ':OUT1_BGEND', kind='config')
    
    #cfd
    out0_cfd_enable = Cpt(EpicsSignal, ':OUT0_CFDEN', kind='config')
    out0_cfd_delay = Cpt(EpicsSignal, ':OUT0_CFDEL', kind='config')
    out0_cfd_threshold = Cpt(EpicsSignal, ':OUT0_CFDTH', kind='config')
    out1_cfd_enable = Cpt(EpicsSignal, ':OUT1_CFDEN', kind='config')
    out1_cfd_delay = Cpt(EpicsSignal, ':OUT1_CFDEL', kind='config')
    out1_cfd_threshold = Cpt(EpicsSignal, ':OUT1_CFDTH', kind='config')
    cfd_intervl = Cpt(EpicsSignal, ':INTRLV_CFDSR', kind='config')
    
    ichan = Cpt(EpicsSignal, ':INTERLEAVE_CHANNEL_RBV', write_pv=':INTERLEAVE_CHANNEL', kind='config')
    config = Cpt(EpicsSignal, ':CONFIG', kind='normal')
    timing_alarm = Cpt(LaserTimingAlarm, '', edge_pv='OUT0_EDGE1', kind='normal')
    avg_channel = Cpt(Signal, value=0, kind='config')
    avg_num = Cpt(Signal, value=100, kind='config')
    avg_num_peaks = Cpt(Signal, value=2, kind='config')

    def __init__(self, prefix, **kwargs):
        self._edge1_sample=deque(maxlen=100)
        self._edge2_sample=deque(maxlen=100)
        self.saved_traces=deque(maxlen=100)
        return super().__init__(prefix, **kwargs)
    
    def avg_config(self, events=None, channel=None, num_peaks=None):
        """
        Setup the QADC averaging parametes

        Parameters
        ----------
        events : int, optional
            Events to average. The default is None.
        channel : int, optional
            Channel to use for averaging (0 or 1). The default is None.
        num_peaks : int, optional
            Number of peaks to look for in averaging. 

        """
        self.unstage()
        d = {}
        if events is not None:
            if isinstance(events, int):
                d.update({'avg_num':events})
            else:
                raise TypeError('events must be of type int')
        if channel is not None:
            if channel in [0,1]: #isinstance(channel, int):
                d.update({'avg_channel':channel})
            else:
                raise TypeError('channel must be 0 or 1')
        if num_peaks is not None:
            if isinstance(num_peaks, int):
                d.update({'avg_num_peaks': num_peaks})
            else:
                raise TypeError('num_peaks must be of type int')
        self.configure(d)
                
    def describe(self):
        det_dict = {self.name + '_EDGE_DIFF_MEAN': {'source':'', 'dtype':'number','shape':[], 'units':'ns'},
                    self.name + '_EDGE1_MEAN': {'source':'', 'dtype':'number','shape':[], 'units':'ns'},
                    self.name + '_EDGE2_MEAN': {'source':'', 'dtype':'number','shape':[], 'units':'ns'},
                    self.name + '_EDGE_DIFF_MEDIAN': {'source':'', 'dtype':'number','shape':[], 'units':'ns'},
                    self.name + '_EDGE1_MEDIAN': {'source':'', 'dtype':'number','shape':[], 'units':'ns'},
                    self.name + '_EDGE2_MEDIAN': {'source':'', 'dtype':'number','shape':[], 'units':'ns'},
                    self.name + '_num_edge1': {'source':'', 'dtype':'number','shape':[], 'units':'None'},
                    self.name + '_num_edge2': {'source':'', 'dtype':'number','shape':[], 'units':'None'}    
                }
        return det_dict
            
    def _configure_qadc(self):
        self.start.put(0)
        self.config.put(1)
        while self.config.get():
            continue
        self.start.put(1)
        
    def stage(self):
        
        def _avg_cb(*args, value, timestamp, **kwargs):
            if isinstance(value, float):
                if (self.avg_num_peaks.get()==1):
                    self._edge1_sample.append(value)
                else:
                    logger.debug('Expected %s edges, but found one edge.' % self.avg_num_peaks.get())
            else:
                if (self.avg_num_peaks.get()==len(value)):
                    self._edge1_sample.append(value[0])  
                    self._edge2_sample.append(value[1])        
                else:
                    logger.debug('Expected %s edges, but found %s edges' % (self.avg_num_peaks.get(), len(value)))
            self._timestamp1=timestamp    
            self._timestamp2=timestamp
                            
        self._edge1_sample=deque(maxlen=self.avg_num.get())
        self._edge2_sample=deque(maxlen=self.avg_num.get())
        
        #stop all previous subscriptions.
        self.out0_edge1.unsubscribe_all()
        self.out0_edge2.unsubscribe_all()
        self.out0_edges.unsubscribe_all()
        self.out1_edge1.unsubscribe_all()
        self.out1_edge2.unsubscribe_all()
        self.out1_edges.unsubscribe_all()
        
        #start new subscriptions
        if self.avg_channel.get()==0:
            self.out0_edges.subscribe(_avg_cb)
        elif self.avg_channel.get()==1:
            self.out1_edges.subscribe(_avg_cb)
        else:
            raise TypeError('self.avg_channel must be 0 or 1.')
        
        return [self]
    
    def trigger(self):
        self._edge1_sample.clear()
        self._edge2_sample.clear()
        
        return DeviceStatus(self, done=True, success=True)
    
    def read(self):
        while (len(self._edge1_sample) < self.avg_num.get()) and (len(self._edge2_sample) < self.avg_num.get()):
            logger.debug('%s/%s'%(len(self._edge1_sample), len(self._edge2_sample)) )
        
        if len(self._edge1_sample)>0:
            mean_edge1 = np.mean(self._edge1_sample)
            median_edge1 = np.median(self._edge1_sample)
        else: 
            mean_edge1 = 0
            median_edge1 = 0
        
        if len(self._edge2_sample)>0:
            mean_edge2 = np.mean(self._edge2_sample)
            median_edge2 = np.median(self._edge2_sample)
        else: 
            mean_edge2 = 0
            median_edge2 = 0
            
        
        diff_mean = mean_edge2-mean_edge1
        diff_median = median_edge2-median_edge1
               
        self.data={self.name + '_EDGE_DIFF_MEAN':{'value': diff_mean,
                                             'timestamp': self._timestamp1 },
                   self.name + '_EDGE_DIFF_MEDIAN':{'value': diff_median,
                                             'timestamp': self._timestamp1 },
                   self.name + '_EDGE1_MEAN':{'value': mean_edge1,
                                             'timestamp': self._timestamp1 },
                   self.name + '_EDGE1_MEDIAN':{'value': median_edge1,
                                             'timestamp': self._timestamp1 },
                   self.name + '_EDGE2_MEAN':{'value': mean_edge2,
                                             'timestamp': self._timestamp2 },
                   self.name + '_EDGE2_MEDIAN':{'value': median_edge2,
                                             'timestamp': self._timestamp2 },
                   self.name + '_num_edge1': {'value': len(self._edge1_sample),
                                              'timestamp': self._timestamp1 },
                   self.name + '_num_edge2': {'value': len(self._edge2_sample),
                                              'timestamp': self._timestamp2 }
                   }
        
        return self.data
    
    def get_diff(self):
        """ Read the average difference in the edges """
        self.trigger()
        return self.read()
    
    def save_raw_traces(self, filename=None, filepath=None):
        """
        save raw qadc (out0 or out1) data to a file. use qadc.avg_config() to select output channel (0 or 1), and number of traces to save.
        The saved traces are also stored in qadc.saved_traces. 

        Parameters
        ----------
        filename : str, optional
            Filename for saving. If None is passed, then a datestring is used.
        filepath : str, optional
            Filepath for saving, the default is tmoopr/qadc_data/.

        """
        def _collect_cb(*args, value, timestamp, **kwargs):
            self.saved_traces.append(value)
            self._timestamp1.append(timestamp)
                      
        self.saved_traces=deque(maxlen=self.avg_num.get())
        self._timestamp1=deque(maxlen=self.avg_num.get())
            
        if self.avg_channel.get()==0:
            self.out0.subscribe(_collect_cb)
        elif self.avg_channel.get()==1:
            self.out1.subscribe(_collect_cb)       
            
        logger.debug('collecting data')
        while (len(self.saved_traces) < self.avg_num.get()):
            logger.debug('%s/%s'%(len(self.saved_traces), self.avg_num.get()))
        logger.debug('data collected')
        
        #stop collection
        self.out0.unsubscribe_all()
        self.out1.unsubscribe_all()
        
        FILEPATH = filepath if filepath is not None else '/cds/home/opr/tmoopr/qadc_data/' 
        if not os.path.exists(FILEPATH):
            os.makedirs(FILEPATH)
        
        filename = filename if filename is not None else 'qadc_out%s_%s.h5' % (self.avg_channel.get(), datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
        
        with h5py.File(FILEPATH + filename, 'w') as hf_dat:
            hf_dat['data'] = self.saved_traces
            hf_dat['timestamps'] = self._timestamp1
            if self.connected:
                for key,value in self.read_configuration().items():
                    hf_dat[key] = [value['value']]
            else:
                logger.warning('Some configurations PVs are not connected, and cannot be saved.')
            
        logger.info('Data saved to %s%s' % (FILEPATH, filename))
            
            
        
    def elog(self, msg=None):
        """
        Write the results of the last measurement to the elog. 

        Parameters
        ----------
        msg : str, optional
            Message to append to the elog. The default is None.

        """
                   
        if hasattr(self, 'data'):
            elog_dict={}
            for key, value in self.data.items():
                elog_dict.update({key: value['value']})
            _log_positions(log=True, msg=msg, curr_positions=elog_dict, tags='coarse_timing qadc')
        else:
            logger.error('No data saved for qadc, try collecting some data.')
    
    def unstage(self):
        self.out0_edge1.unsubscribe_all()
        self.out0_edge2.unsubscribe_all()
        self.out0_edges.unsubscribe_all()
        self.out1_edge1.unsubscribe_all()
        self.out1_edge2.unsubscribe_all()
        self.out1_edges.unsubscribe_all()
        self.out0.unsubscribe_all()
        self.out1.unsubscribe_all()
        return[self]   

class LaserSpectrometer(BaseInterface, Device):
    tab_component_names = True
    tab_whitelist = ['plot']
    
    spectrum = Cpt(EpicsSignalROStop,':SPECTRUM', kind='normal')
    status = Cpt(EpicsSignalROStop, ':STATUS', kind='normal')
    exposure_time = Cpt(EpicsSignal, ':GET_EXPOSURE_TIME', write_pv=':SET_EXPOSURE_TIME', kind='config')
    exposures_to_avg = Cpt(EpicsSignal, ':SET_AVG_CNT', kind='config')
    scan_rate = Cpt(EpicsSignal, ':SOFT_TRIGGER_MODE', kind='config')
    trig_enable = Cpt(EpicsSignal,':SET_TRIG_ENABLE', kind='config')
    temp = Cpt(EpicsSignalRO, ':TEMP', kind='omitted')
    settle_time = Cpt(Signal, value=100E-3, kind='omitted', 
                      doc='time to wait after spectrometer reports idle to read.')
    
    def describe(self):
        det_dict = {self.spectrum.name: {'source':self.spectrum.pvname, 'dtype':'array', 'shape':[2500], 'units':'None'}}
        return det_dict
    
    def stage(self):
        return [self]
    
    def unstage(self):
        return [self]
    
    def trigger(self):
        def check_collecting(*, old_value, value, timestamp, **kwargs):
            logger.debug('old value: %s, new value: %s, timestamp: %s' % (old_value, value, timestamp))
            return (old_value==0 and value==2) #old value idle, new value collecting
        
        def check_finished(*, old_value, value, timestamp, **kwargs):
            logger.debug('old value: %s, new value: %s, timestamp: %s' % (old_value, value, timestamp))
            return (old_value>0 and value==0) #old value collecting, new value idle
        
        self.spectrum.trigger()
        status = SubscriptionStatus(self.status, check_collecting, timeout=self.spectrum.timeout)
        while not status.done: continue
        if not status.success: raise Exception('Timeout')
    
        status = SubscriptionStatus(self.status, check_finished, timeout=self.spectrum.timeout, settle_time=self.settle_time.get())    
        return status
    
    def read(self):
        result = self.spectrum.read()
        logger.debug('timestamp: %s' % (result[self.spectrum.name]['timestamp']))
        return result
    
    def plot(self):
        status = self.trigger()
        status.wait()
        result = self.read()
        plt.figure(483)
        plt.plot(result[self.spectrum.name]['value'])
    
    
class TmoLaserIP(BaseInterface, Device):
    """
    Base class for TMO laser controls
    """
    tab_component_names= True
    tab_whitelist = ['summary', 'shutter_open', 'shutter_close', 'block_atm', 'enable_atm']
    
    ejx_shutter = None
    atm_shutter = None
    
    def shutter_close(self):
        """ Block the laser at the IP by closing the EJX shutter"""
        return self.ejx_shutter.close()
        
    def shutter_open(self):
        """ Open the IP laser shutter."""
        return self.ejx_shutter.open()
        
    def block_atm(self):
        """ Block the ATM beam using the ATM shutter."""
        return self.atm_shutter.close()
        
    def enable_atm(self):
        """ Open ATM shutter.""" 
        return self.atm_shutter.open()
    
    def summary(self, log=False, msg=None, **kwargs):
        """ Check the status of laser devices."""
        curr_positions= {}
        for key in dir(self):
            motor = getattr(self, key)
            if hasattr(motor, 'connected'):
                if 'Ell' in str(type(motor)):
                    curr_positions[key] = getattr(motor, 'position')        
                elif motor.connected:
                    if 'SmarAct' in str(type(motor)):
                        if 'TipTilt' in str(type(motor)):
                            if 'Encoded' in str(type(motor)):
                                curr_positions[(key+'_tip')]=motor.tip.position
                                curr_positions[(key+'_tilt')]=motor.tilt.position
                            else:
                                curr_positions[(key+'_tip')]=motor.tip.total_step_count.get()
                                curr_positions[(key+'_tilt')]=motor.tilt.total_step_count.get()
                        elif ('Dual' in str(type(motor))) or ('Three' in str(type(motor))):
                            for keys in dir(motor):
                                mot = getattr(motor, keys)
                                if hasattr(mot, 'position'):
                                    position = getattr(mot, 'position')
                                    if isinstance(position, float):
                                        curr_positions[(key+'_'+keys)] = position
                        elif 'OpenLoop' in str(type(motor)):
                            curr_positions[key] = motor.total_step_count.get()
                        else:
                            curr_positions[key] = motor.position    
                    elif 'SmarPod' in str(type(motor)):
                        for keys in dir(motor):
                            axis = getattr(motor, keys)
                            if hasattr(axis, 'position'):
                                position = getattr(axis, 'position')
                                curr_positions[key+'_'+keys] = position
                    elif hasattr(motor, 'position'):
                        curr_positions[key] = getattr(motor, 'position')
        kwargs['tags'] = _add_tags(kwargs.get('tags', ''),'laser')                            
        _log_positions(curr_positions, log, msg=msg, **kwargs)
    
class TmoLaserIP1(TmoLaserIP):
    """
    This class controls motors associated with the IP1 laser system in TMO.
    stop method closes the laser shutter
    enable method will open the laser shutter
    """
    tab_component_names= True
#    tab_whitelist = ['shutter_open', 'shutter_close', 'block_atm', 'enable_atm']
        
#    lens_x = Cpt(SmarAct, 'LM1K4:EJX_MP2_LS1_LM2', kind='normal', 
#                 doc='LM1K4 lens motion x')
#    lens_y = Cpt(SmarAct, 'LM1K4:EJX_MP2_LS1_LM3', kind='normal', 
#                 doc='LM1K4 lens motion y')
#    lens_z = Cpt(SmarAct, 'LM1K4:EJX_MP2_LS1_LM1', kind='normal', 
#                 doc='LM1K4 lens motion z')
#    lens = Cpt(SmarActThreeAxis, 'LM1K4:EJX_MP2_LS1', kind='normal',
#                 doc='LM1K4 lens')
    inj_mr = Cpt(SmarAct, 'LM1K4:INJ_DP2_MR1', kind='normal', 
                 doc='')
    atm_mr3_tt = Cpt(SmarActTipTilt, prefix='LM1K4:ATM_MP1_MR3', tip_pv="_TIP1", tilt_pv="_TILT1", kind='normal',
                 doc='')
    oap_tt1 = Cpt(SmarActEncodedTipTilt, prefix="LM1K4:EJX_MP1_OAP1_MR1", tip_pv="_TIP1", tilt_pv="_TILT1", kind='normal',
                 doc='LM1K4 focusing optic 1 tip/tilt')
    oap_z = Cpt(SmarAct, 'LM1K4:EJX_MP1_OAP1_LM1', kind='normal', 
                 doc='LM1K4 focusing optics 2, Z-motion')
    oap_tt3 = Cpt(SmarActEncodedTipTilt, prefix="LM1K4:EJX_MP1_OAP1_MR3", tip_pv="_TIP1", tilt_pv="_TILT1", kind='normal',
                 doc='LM1K4 focusing optic 3 tip-tilt')
    wp_dp2 = Cpt(EllRotation, 'LM1K4:EJX:ELL:1', channel=1, port=0, kind='normal',
                 doc='')
    ejx_qwp_1 = Cpt(EllRotation, 'LM1K4:EJX:ELL:2', channel=1, port=0, kind='normal', atol = 0.05, 
                 doc='EJX waveplate to adjust elliptical polarization')
    ejx_qwp_2 = Cpt(EllRotation, 'LM1K4:EJX:ELL:2', channel=2, port=0, kind='normal', atol = 0.05, 
                 doc='EJX waveplate')
    ejx_wp_3 = Cpt(EllRotation, 'LM1K4:EJX:ELL:4', channel=1, port=0, kind='normal', atol = 0.05,
                   doc='LM3K4 rotation stage')
    ejx_slider = Cpt(Ell9, 'LM1K4:EJX:ELL:1', channel=3, port=0, kind='normal',
                 doc='ejx slider')
    atm_qwp = Cpt(EllRotation, 'LM1K4:ATM:ELL:1', channel=3, port=0, kind='normal', 
                 doc='QWP in ATM arm')
    att_wp = Cpt(SmarAct, 'LM1K4:INJ_MP1_ATT1_WP1', kind='normal', 
                 doc='wavelplate in a INJ attenuator')    
    li3k4 = Cpt(SmarActDual, 'LM1K4:EJX_DP6_CVS1', kind='normal', 
                 doc='LI3K4 outcoupling mirrors')
    ejx_shutter = Cpt(LaserShutter, 'LM1K4:EJX', kind='normal', 
                 doc='Laser shutter in EJX tile')
    atm_shutter = Cpt(LaserShutter, 'LM1K4:ATM', kind='normal', 
                 doc='shutter for ATM arm')
    ejx_ap2 = Cpt(SmarActOpenLoop, 'LM1K4:EJX_DP2_AP2', kind='normal',
                     doc='Remote Iris on EJX.')
#    timing_alarm = Cpt(LaserTimingAlarm, 'LM1K4:QADC:01', edge_pv='OUT0_EDGE1', kind='normal')
    lm1k4_qadc = Cpt(TmoQadc134, 'LM1K4:QADC:01', kind='normal')
    lm1k4_spec = Cpt(LaserSpectrometer, 'LM1K4:EJX_DP6_TF1_SP1', kind='normal')
        
class TmoLaserIP2(TmoLaserIP):
    """
    This class controls motors associated with the IP2 laser system in TMO.
    """
    tab_component_names= True
    #tab_whitelist = ['shutter_open', 'shutter_close', 'block_atm', 'enable_atm']
    
#    smpd = Cpt(SmarPod, 'TMO:SPD:01', kind='normal',
#                 doc='IP2 SmarPod for OAP')

    lens_x = Cpt(SmarAct, 'LM2K4:EJX_MP2_LS1_LM2', kind='normal', 
                 doc='LM2K4 lens motion x')
    lens_y = Cpt(SmarAct, 'LM2K4:EJX_MP2_LS1_LM3', kind='normal', 
                 doc='LM2K4 lens motion y')
    lens_z = Cpt(SmarAct, 'LM2K4:EJX_MP2_LS1_LM1', kind='normal', 
                 doc='LM2K4 lens motion z')    
    smpd = Cpt(SmarPodMotor, 'TMO:DREAM:SMARPOD', kind='normal',
                 doc='IP2 SmarPod for OAP with missing axis')
    att_wp = Cpt(SmarAct, 'LM2K4:INJ_MP1_ATT1_WP1', kind='normal', 
                 doc='wavelplate in a INJ attenuator')  
    ejx_mr_z = Cpt(SmarAct, 'LM2K4:EJX_MP1_MR99_LM1', kind='normal',
                 doc='mirror translation stage before chamber.')
    ejx_mr_tt = Cpt(SmarActEncodedTipTilt, prefix="LM2K4:EJX_MP1_MR99", tip_pv="_TIP1", tilt_pv="_TILT1", kind='normal',
                 doc='mirror tip-tilt before chamber.')
    atm_qwp_1 = Cpt(EllRotation, 'LM2K4:ATM:ELL:1', channel=1, port=0, kind='normal', atol = 0.05, 
                 doc='QWP in ATM to adjust interferometric mode')
    atm_shutter = Cpt(LaserShutter, 'LM2K4:ATM', kind='normal', 
                 doc='shutter for ATM arm')
    atm_mr_tt = Cpt(SmarActEncodedTipTilt, prefix="LM2K4:ATM_MP1_MR3", tip_pv="_TIP1", tilt_pv="_TILT1", kind='normal',
                 doc='mirror tip-tilt for atm.')
    atm_mic_tt = Cpt(SmarActEncodedTipTilt, prefix="LM2K4:ATM_MP1_IMG1_BS1", tip_pv="_TIP1", tilt_pv="_TILT1", kind='normal',
                 doc='atm microscope beam splitter tip-tilt.')
    ejx_shutter = Cpt(LaserShutter, 'LM2K4:EJX', kind='normal', 
                 doc='shutter for EJX')
    ejx_slider_ic = Cpt(Ell9, 'LM2K4:EJX:ELL:1', channel=1, port=0, kind='normal',
                 doc='ejx incoupling slider')
    ejx_slider_oc = Cpt(Ell9, 'LM2K4:EJX:ELL:1', channel=2, port=0, kind='normal',
                 doc='ejx outcoupling slider')
    pp_delay = Cpt(SmarAct, 'LM2K4:HRM_MP0_DLY1', kind='normal',
                   doc='pump-probe delay stage')
    ejx_frog = Cpt(SmarAct, 'LM2K4:EJX_DP1_DLY1', kind='normal',
                   doc='FROG delay stage in ejx tile.')
    ejx_diag = Cpt(SmarAct, 'LM2K4:EJX_DP2_MR1_LM1', kind='normal',
                   doc='mirror for diagnostics')
#    timing_alarm = Cpt(LaserTimingAlarm, 'LM2K4:QADC:01', edge_pv='OUT0_EDGE1', kind='normal')
    lm2k4_qadc = Cpt(TmoQadc134, 'LM2K4:QADC:01', kind='normal')
      
    
class TmoLaser(BaseInterface, Device):
    """
    Description
    """
    tab_component_names= True
    tab_whitelist = ['open_ip1_shutter', 'close_ip1_shutter','open_ip2_shutter', 'close_ip2_shutter']
    
    ip1 = Cpt(TmoLaserIP1,'',kind='normal')
    ip2 = Cpt(TmoLaserIP2,'',kind='normal')
    
    def open_ip1_shutter(self):
        return self.ip1.shutter_open()
    
    def close_ip1_shutter(self):
        return self.ip1.shutter_close()
    
    def open_ip2_shutter(self):
        return self.ip2.shutter_open()
    
    def close_ip2_shutter(self):
        return self.ip2.shutter_close()
  
#tmo_laser = TmoLaserIP1(name='tmo_laser')
#stop_laser = tmo_laser.shutter_close

tmo_laser = TmoLaser(name='tmo_laser')
stop_laser = tmo_laser.close_ip1_shutter

def get_laser_ip1(log=False, msg=None, include_cam=None, **kwargs):
    """
    Get the current position of the IP1 laser motors
    """
    kwargs['tags'] = _add_tags(kwargs.get('tags', ''),'ip1')
    tmo_laser.ip1.summary(log, msg, include_cam=include_cam, **kwargs)
get_laser_ip1.__doc__ += _log_positions.__doc__
    
def get_laser_ip2(log=False, msg=None, include_cam=None, **kwargs):
    """
    Get the current position of the IP2 laser motors
    """
    kwargs['tags'] = _add_tags(kwargs.get('tags', ''),'ip2')
    tmo_laser.ip2.summary(log, msg, include_cam=include_cam, **kwargs)
get_laser_ip2.__doc__ += _log_positions.__doc__

def get_smrpd(log=False, msg=None, include_cam=None, **kwargs):
    """
    Get the current position of the IP2 smartpod motors
    """
    curr_positions=OrderedDict()
    for keys in dir(tmo_laser.ip2.smrpd):
        axis = getattr(tmo_laser.ip2.smrpd, keys)
        if hasattr(axis, 'position'):
            position = getattr(axis, 'position')
            curr_positions[tmo_laser.ip2.smrpd.name+'_'+keys] = position
    kwargs['tags'] = _add_tags(kwargs.get('tags', ''),'smrpd')        
    _log_positions(curr_positions, log, msg=msg, **kwargs)
    
get_smrpd.__doc__ += _log_positions.__doc__

def get_lm2k4(log=False, **kwargs):
    """
    Get current position of the laser optics associated with lm2k4.
    Parameters:
    --------------- 
    log - boolean to log the positions in the elog 
    msg - optional argument to append a message to the elog with the positions 
    """
    curr_positions= {
    "lens X":tmo_laser.lens_x.position, 
    "lens Y":tmo_laser.lens_y.position,
    "lens Z":tmo_laser.lens_z.position,
#       "WP":tmo_laser.pol_wp.position,
    }
    kwargs['tags'] = _add_tags(kwargs.get('tags', ''),'li2k4')
    _log_positions(curr_positions, log, **kwargs)
  
def get_lm1k4(log=False, **kwargs):
    """
    Get current position of the laser optics associated with lm1k4.
    Parameters:
    --------------- 
    log - boolean to log the positions in the elog 
    msg - optional argument to append a message to the elog with the positions 
    """
    curr_positions= {
        "OAP1 tip":tmo_laser.oap_tt1.tip.total_step_count.get(), 
       "OAP1 tilt":tmo_laser.oap_tt1.tilt.total_step_count.get(),
        "OAP3 tip":tmo_laser.oap_tt3.tip.total_step_count.get(),
       "OAP3 tilt":tmo_laser.oap_tt3.tilt.total_step_count.get(),
           "OAP Z":tmo_laser.oap_z.position,
#    "LM1K4 Mirror":li1k4.mirror.position,
#     "LM1K4 Motor":li1k4.y_motor.position
    }
    for keys in dir(li1k4):
        motor = getattr(li1k4, keys)
        if hasattr(motor, "position"):
            curr_positions[motor.name] = motor.position
    kwargs['tags'] = _add_tags(kwargs.get('tags', ''), 'li1k4')
    _log_positions(curr_positions, log, **kwargs)

def get_atm(log=False, **kwargs):
    """
    Get current positions of the ATM optics.
    
    Parameters:
    ---------------
    log - booleam to log the positions in the elog
    msg - optional argument to append a message to the elog with the positions. 
    """
    curr_positions={
     "ATM MR tip":tmo_laser.atm_mr3_tt.tip.total_step_count.get(),
    "ATM MR tilt":tmo_laser.atm_mr3_tt.tilt.total_step_count.get(),
    "ATM target":tm1k4.target.position,
    "ATM y-Motor":tm1k4.y_motor.position
    }
    for keys in dir(tm1k4):
        motor = getattr(tm1k4, keys)
        if hasattr(motor, "position"):
            curr_positions[motor.name] = motor.position           
    kwargs['tags'] = _add_tags(kwargs.get('tags', ''),'atm')
    _log_positions(curr_positions, log, **kwargs)
  

class Wav8Fex(BaseInterface, Device):
    tab_component_names= True
    
    #Integral parameters
    IntegralSize = Cpt(EpicsSignal, ':Top:Integrators:IntegralSize_RBV', write_pv=':Top:Integrators:IntegralSize', kind='normal')
    IntegralSigned = Cpt(EpicsSignal, ':Top:Integrators:IntegralSigned_RBV', write_pv=':Top:Integrators:IntegralSigned', kind='normal')
    BaselineSize = Cpt(EpicsSignal, ':Top:Integrators:BaselineSize_RBV', write_pv=':Top:Integrators:BaselineSize', kind='normal')
    TrigDelay = Cpt(EpicsSignal, ':Top:Integrators:TrigDelay_RBV', write_pv=':Top:Integrators:TrigDelay', kind='normal')
    
    #Integral Readback
    adc_integral0 = Cpt(EpicsSignalRO,':Top:Integrators:AdcIntegral[0]', kind='hinted')
    adc_integral1 = Cpt(EpicsSignalRO,':Top:Integrators:AdcIntegral[1]', kind='hinted')
    adc_integral2 = Cpt(EpicsSignalRO,':Top:Integrators:AdcIntegral[2]', kind='hinted')
    adc_integral3 = Cpt(EpicsSignalRO,':Top:Integrators:AdcIntegral[3]', kind='hinted')
    adc_integral4 = Cpt(EpicsSignalRO,':Top:Integrators:AdcIntegral[4]', kind='hinted')
    adc_integral5 = Cpt(EpicsSignalRO,':Top:Integrators:AdcIntegral[5]', kind='hinted')
    adc_integral6 = Cpt(EpicsSignalRO,':Top:Integrators:AdcIntegral[6]', kind='hinted')
    adc_integral7 = Cpt(EpicsSignalRO,':Top:Integrators:AdcIntegral[7]', kind='hinted')
    
    def stage(self):
        """"Needed for future use"""
        return [self]
    
    def trigger(self):
        for key in dir(self):
            if 'adc_int' in key:
                adc_int = getattr(self, key)
                adc_int.trigger()
        return DeviceStatus(self, done=True, success=True)
    
    def read(self):
        self.data=OrderedDict()
        for key in dir(self):
            if 'adc_int' in key:
                adc_int = getattr(self, key)
                self.data.update(adc_int.read())
        return self.data
    
    def unstage(self):
        """"Needed for future use"""
        return [self]
    
    def describe(self):
        data = OrderedDict()
        for key in dir(self):
            if 'adc_int' in key:
                adc_int = getattr(self, key)
                #data[key] = {'source':adc_int.pvname, 'dtype':'number','shape':[], 'units':'adu'}
                data.update(adc_int.describe())
        return data

tmo_las_ip1_wav8 = Wav8Fex('LM1K4:W8:04', name='tmo_laser_ip1_wav8') 
tmo_las_ip2_wav8 = Wav8Fex('LM2K4:W8:01', name='tmo_laser_ip2_wav8') 

def tmo_laser_coarse_timing_scan(motor, scan_range, nevents_step, ch, avg_type='MEDIAN'):
    """
    Here is a function to scan a motor over a range specified by scan_range. The results are plotted and fit with a line.
    This was a suggestion from Mat Britton for a way to find timing. 

    Parameters
    ----------
    motor : motor device
        Motor to scan (lxt or txt).
    scan_range : list of float
        Range to scan txt.
    nevents_step : int
        Number of points to average.
    ch : int
        Channel of qadc to use for averaging (0 or 1).

    Returns
    -------
    Fit result : Dictoinary
    
    """
    #Define the function to fit to the data
    def line(x, a, b):
        #return a*x+b
        return -abs(a*x+b)
    #create an lmfit model for the fit with an initial guess
    model = lmfit.Model(line)
    init_guess = {'a':1E9, 'b':0}
    
    #Figure out which detector to use
    if 'ip1' in motor.name:
        det = tmo_laser.ip1.lm1k4_qadc
        x_str = motor.name + '_delay'
    elif 'ip2' in motor.name:
        det = tmo_laser.ip2.lm2k4_qadc
        x_str = motor.name + '_delay'
    else: #lxt is passed
        det = tmo_laser.ip1.lm1k4_qadc
        x_str = motor.name + '_setpoint'
    
    #define strings for plotting functions
    det_str = ('%s_EDGE_DIFF_%s' % (det.name, avg_type.upper()) )
    
    #configure the qadc for the correct channel and number of points to average
    det.avg_config(events=nevents_step, channel=ch)
    
    #configure a LiveFit object to do the plotting and fitting
    lf = LiveFit(model, det_str, {'x':x_str}, init_guess)
        
    #configure plots for the data
    fig, ax = plt.subplots() #create figure
    lfp = LiveFitPlot(lf, ax=ax, color='r')
    lp = LivePlot(det_str, x_str, ax=ax, marker='o', linestyle='none')
    
    #Use blue sky to do the scan and fit. 
    RE( bp.list_scan([det], motor, scan_range), [lp, lfp])
       
    return lf.result


"""
    lcls_base_rate=928571.4286
    laser_base_rate=carbide_pp_tpr.eventrate.get()/0.98
    laser_divisor = round(lcls_base_rate/laser_base_rate)
    goose_rate=goose_tpr.eventrate.get()/0.98
    goose_divisor = round(lcls_base_rate/goose_rate)
    overlap_divisor = np.lcm(laser_divisor,goose_divisor)
    overlap_percent = laser_divisor/goose_divisor
    goose_delay_ticks = goose_tpr.delay_ticks.get() - carbide_pp_tpr.delay_ticks.get()
    goose_delay = goose_delay_ticks * (1/185.714286E-3)
    
    timingDict = {'Base Rate':laser_base_ra""te,
                  'Base Sequence Code': str(carbide_pp_tpr.seqcode.get()),
                  'Goose Rate':goose_rate,
                  'Goose Sequence Code': str(goose_tpr.seqcode.get()),
                  'Goose Delay [ns]':goose_delay,
                  'Laser/Goose Overlap':1/overlap_percent
                  }
    kwargs['tags'] = kwargs.get('tags', ' ') + ' laser_trigger'
    _log_positions(timingDict, log, **kwargs)
    
"""

