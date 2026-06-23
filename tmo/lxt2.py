from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from pcdsdevices.device import (Component as Cpt, ObjectComponent as OCpt)
from ophyd import FormattedComponent as FCpt
from pcdsdevices.lxe import LaserTiming, TimeToolDelay
from pcdsdevices.lxe import Lcls2LaserTiming
from pcdsdevices.signal import MultiDerivedSignal, MultiDerivedSignalRO
from pcdsdevices.type_hints import SignalToValue
from pcdsdevices.tpr import TprTrigger, TimingMode
from pcdsdevices.epics_motor import SmarAct
from pcdsdevices.pseudopos import SyncAxis, delay_instance_factory, delay_class_factory
from ophyd.signal import EpicsSignalRO, EpicsSignal
from ophyd.status import MoveStatus, DeviceStatus, SubscriptionStatus
from tmo.tmo_utils import log_positions as _log_positions
from elog import HutchELog as elog_temp
elog = elog_temp.from_registry()
import logging
logger = logging.getLogger(__name__)
from tmo.db import archive
import datetime
from collections import OrderedDict
import numpy as np
import pandas as pd
import sys
from dev.devpath.hutch_python.log_setup import debug_mode

############
#If the laser changes edit the following variables. 
TMO_LASER_TIME_PV='LAS:LHN:LLG2:01'  #Zone 3
TMO_LASER_TPR_PV='LAS:LHN:TPR:01'    #Zone 3
#TMO_LASER_TIME_PV='LAS:LHN:LLG2:02'  #Zone 2
#TMO_LASER_TPR_PV='LAS:LHN:TPR:02'    #Zone 2
############

#######
#LCLS-1
#lxt = LaserTiming('LAS:FS14', name='lxt') #this is for lcls1

#######
#LCLS-2
#lxt = Lcls2LaserTiming(TMO_LASER_TIME_PV, name='lxt') #this is for lcls2

#######
#txt = delay_instance_factory('LM1K4:COM_MP2_DLY1', motor_class=SmarAct,
#                             egu='s', n_bounces=16, name='txt')

class Lcls2LaserTimingLog(Lcls2LaserTiming):
    
    def get_timing(self):
        timing_dict = OrderedDict()
        timing_dict[self.name+" pos [ps]"]=self.wm()*1e12
        timing_dict[self.name+" offset [ns]"]=self.user_offset.get()*1e9
        timing_dict[self.name+" total delay [ns]"]=(self.user_offset.get() - self.setpoint.get()) *1e9 
        return timing_dict
    
    def set_current_position(self, position):
        old_offset = self.user_offset.get()
        super().set_current_position(position)
        logging_dict = self.get_timing()
        new_offset = self.user_offset.get()
        msg=self.name+('.set_current_position(%s). Offset moved from %s to %s' %(position, old_offset,new_offset))
        _log_positions(logging_dict, log=True, msg=msg, tags='fs_timing sctipt-generated')

lxt = Lcls2LaserTimingLog(TMO_LASER_TIME_PV, name='lxt')

class SmarActDelayLog(delay_class_factory(SmarAct)):
    
    target_to_zero_voltage = Cpt(EpicsSignal, ':TTZV_MODE_RBV', write_pv=':TTZV_MODE')
    state = Cpt(EpicsSignalRO, ':STATE_RBV')
    
    def __init__(self, prefix, **kwargs):
        ret = super().__init__(prefix, **kwargs) 
        #here is where we should add the move checker.
        return ret
        
    def get_timing(self):
        timing_dict = OrderedDict()
        timing_dict[self.name+" pos [ps]"] = self.wm()*1e12
        timing_dict[self.name+" offset [ns]"] = self.user_offset.get() * 1E9
        timing_dict[self.name+" total delay [ns]"] = (self.user_offset.get() - self.delay.setpoint.get()) * 1E9
        timing_dict[self.name+" user stage [mm]"] = self.motor.position # txt.get()[2][0]
        return timing_dict
    
    def set_current_position(self, position):
        old_offset = self.user_offset.get()
        super().set_current_position(position)
        logging_dict = self.get_timing()
        new_offset = self.user_offset.get()
        msg=self.name+('.set_current_position(%s).\nOffset moved from %s to %s' %(position, old_offset,new_offset))
        _log_positions(logging_dict, log=True, msg=msg, tags='fs_timing sctipt-generated')
            
    def stage(self):
        self.target_to_zero_voltage.put(0)
        return super().stage()
    
    def unstage(self):
        self.target_to_zero_voltage.put(1)
        return super().unstage()
    
    def move(self, position, moved_cb=None, timeout=None, wait=None):
        
        def check_done(*, value, **kwargs):
             "Mark status as finished when both done moving (B0 == 0) and in position (B13 == 1)"
             bit_str = bin(value)[2:][::-1] #write result as bit string, flip
             B0 = int(bit_str[0])
             B13 = int(bit_str[18]) #18 s B13..... Adam says
             return ( (B0==0) and (B13==1) )

        super().move(position, moved_cb=moved_cb, timeout=timeout, wait=wait)
        status = SubscriptionStatus(self.state, check_done, timeout=timeout,
                             settle_time=self._settle_time)
        return status
        

class TmoTxt(BaseInterface, Device):
    """
    TMO ATM delay stage control. There are two sub-processes ip1 and ip2. 
    Both are 16 bounce stages. 
    """
    tab_component_names = True
    
#    ip1 = FCpt(SmarActDelayLog, 'LM1K4:COM_MP2_DLY1', egu='s', n_bounces=16, invert=False, kind='hinted')
    #4/12/2026 we noticed the handed-ness of txt has changed (mabye the stage was changed). So we will invert.
    ip1 = FCpt(SmarActDelayLog, 'LM1K4:COM_MP2_DLY1', egu='s', n_bounces=16, invert=True, kind='hinted')

#    ip2 = FCpt(SmarActDelayLog, 'LM2K4:COM_MP2_DLY1', egu='s', n_bounces=16, invert=True, kind='hinted')
    #On 10/1/2025 we observed opposite behavior of txt and lxt and so we changed the invert. 
    ip2 = FCpt(SmarActDelayLog, 'LM2K4:COM_MP2_DLY1', egu='s', n_bounces=16, invert=False, kind='hinted')

txt = TmoTxt('',name='txt')    

#old version of txt define for both IPs    
#txt_1 = SmarActDelayLog('LM1K4:COM_MP2_DLY1', egu='s', n_bounces=16, invert=False, name='txt_1')
#txt_2 = SmarActDelayLog('LM2K4:COM_MP2_DLY1', egu='s', n_bounces=16, invert=True, name='txt_2')

class LXTTTC(SyncAxis):
    lxt = OCpt(lxt)
    tab_component_names = True
    scales = {'lxt': 1, 'txt': 1} 
    warn_deadband = 5e-14
    fix_sync_keep_still = 'lxt'
    sync_limits = (-10e-6, 10e-6)

    def get_timing(self):
        timing_dict = OrderedDict()
        timing_dict["lxt_ttc pos [ps]"]=self.wm()*1e12
        if lxt.connected:
            timing_dict.update(self.lxt.get_timing())
        else:
            logger.error('lxt not connected!')
        if self.txt.connected:
            timing_dict.update(self.txt.get_timing())
        else:
            logger.error('txt not connected!')        
        return timing_dict
    
    def set_current_position(self, position):
        old_offset = self.user_offset.get()
        super().set_current_position(position)
        logging_dict = self.get_timing()
        new_offset = self.user_offset.get()
        msg=self.name+('.set_current_position().\nOffset moved from %s to %s' %(old_offset,new_offset))
        _log_positions(logging_dict, log=True, msg=msg, tags='fs_timing sctipt-generated')

class LXTTTC1(LXTTTC):
    tab_component_names = True
#    txt = OCpt(txt_1)
    txt = OCpt(txt.ip1)
    
class LXTTTC2(LXTTTC):
    tab_component_names = True
#    txt = OCpt(txt_2)
    txt = OCpt(txt.ip2)

lxt_ttc_ip1 = LXTTTC1('', name='lxt_ttc_ip1')
lxt_ttc_ip2 = LXTTTC2('', name='lxt_ttc_ip2')

class TmoLxtTtcCpt(BaseInterface, Device):
    tab_component_names = True
    
    ip1 = Cpt(LXTTTC1, '', kind='normal')
    ip2 = Cpt(LXTTTC2, '', kind='normal')
    
class TmoLxtTtcOCpt(BaseInterface, Device):
    tab_component_names = True
 
    ip1 = OCpt(lxt_ttc_ip1)
    ip2 = OCpt(lxt_ttc_ip2)

lxt_ttc = TmoLxtTtcOCpt('', name='lxt_ttc')
#lxt_ttc = TmoLxtTtcCpt('', name='lxt_ttc')
        

#class TmoLxt(Lcls2LaserTiming):
#    ttc = Cpt(LXTTTC, '')    
#lxt2 = TmoLxt('LAS:LHN:LLG2:01', name='lxt')

######
#Temporary fix to understand lxt_ttxc issues.
######
# class LXTTTC_Debug(SyncAxis):
#     lxt = OCpt(lxt)
#     txt = OCpt(txt.ip2)
#     tab_component_names = True
#     scales = {'txt': 1} #changed by JPC 11/04/24
#     warn_deadband = 5e-14
#     fix_sync_keep_still = 'lxt'
#     sync_limits = (-10e-6, 10e-6)

#     def get_timing(self):
#         timing_dict = OrderedDict()
#         timing_dict["lxt_ttc pos [ps]"]=self.wm()*1e12
#         if lxt.connected:
#             timing_dict.update(self.lxt.get_timing())
#         else:
#             logger.error('lxt not connected!')
#         if self.txt.connected:
#             timing_dict.update(self.txt.get_timing())
#         else:
#             logger.error('txt not connected!')        
#         return timing_dict
    
#     def set_current_position(self, position):
#         old_offset = self.user_offset.get()
#         super().set_current_position(position)
#         logging_dict = self.get_timing()
#         new_offset = self.user_offset.get()
#         msg=self.name+('.set_current_position().\nOffset moved from %s to %s' %(old_offset,new_offset))
#         _log_positions(logging_dict, log=True, msg=msg, tags='fs_timing sctipt-generated')

# lxt_ttc_ip2 = LXTTTC_Debug('', name='lxt_ttc_ip2')



######


def shift_t0(shift):
        """
        This function shifts t0 at the interaction point by the amount passed (in seconds).
        It leaves the position of lxt_ttc unchanged.
        """
        get_timing(log=True, msg='Timing  before adjusting lxt.')
#        curr = lxt_ttc_1.wm()
        curr = lxt_ttc.ip1.wm()        
        lxt.mvr(shift)
#        lxt_ttc_1.set_current_position(curr)
        lxt_ttc.ip1.set_current_position(curr)
#        txt_position = txt_1.motor.user_readback.get()
        txt_position = txt.ip1.motor.user_readback.get()
        msg = ("moved lxt offset by %s ps to compensate for drift. Current position of txt stage for lxt_ttc=%s: %s"  % (shift*1E12, curr, txt_position) )
        get_timing(log=True, msg=msg)
        return txt_position

def tweak_t0(shift):
        """
        This function is used to shift t0 at the interaction point. It opens a tweak function with arguement shift.
        It leaves the position of lxt_ttc unchanged.
        """
        get_timing(log=True, msg='Timing  before adjusting lxt.')
#        curr = lxt_ttc_1.wm()
        curr = lxt_ttc.ip1.wm()
        lxt.tweak(shift)
#        lxt_ttc_1.set_current_position(curr)
        lxt_ttc.ip1.set_current_position(curr)
#        txt_position = txt_1.motor.user_readback.get()
        txt_position = txt.ip1.motor.user_readback.get()
        msg = ("moved lxt offset by %s ps to compensate for drift. Current position of txt stage for lxt_ttc=%s: %s"  % (shift*1E12, curr, txt_position) )
        get_timing(log=True, msg=msg)
        return


def get_timing(log=False, msg=None, **kwargs):
    """
    Get current positions of laser timing variables.
    log - boolean to log the positions in the elog 
    msg - optional argument to append a message to the elog with the KB positions 
    """
    curr_positions = OrderedDict()
    curr_positions["lxt pos [ps]"] = None
    curr_positions["txt_ip1 pos [ps]"] = None
    curr_positions["txt_ip2 pos [ps]"] = None

    """    
    if lxt.connected:
        curr_positions["lxt pos [ps]"]=lxt.wm()*1e12
        curr_positions["lxt_ttc pos [ps]"]=lxt_ttc.wm()*1e12
        curr_positions["lxt offset [ns]"]=lxt.user_offset.get()*1e9 #get()[3]*1e9
        curr_positions["lxt total delay [ns]"]=(lxt.user_offset.get() - lxt.setpoint.get()) *1e9 #(lxt.get()[3] - lxt.get()[0]) * 1e9, 
    else:
        curr_positions["lxt pos [ps]"]='n/c'
        curr_positions["lxt_ttc pos [ps]"]='n/c'
        curr_positions["lxt offset [ns]"]='n/c'
        curr_positions["lxt total delay [ns]"]='n/c'
        logger.error('lxt not connected!')
    
    if txt.connected:
        curr_positions["txt pos [ps]"] = txt.wm()*1e12
        curr_positions["txt offset [ns]"] = txt.user_offset.get() * 1E9
        curr_positions["txt total delay [ns]"] = (txt.user_offset.get() - txt.delay.setpoint.get()) * 1E9
        curr_positions["txt user stage [mm]"] = txt.motor.position # txt.get()[2][0]
    else:
        curr_positions["txt pos [ps]"] = 'n/c'
        curr_positions["txt offset [ns]"] = 'n/c'
        curr_positions["txt total delay [ns]"] = 'n/c'
        curr_positions["txt user stage [mm]"] = 'n/c'
        logger.error('txt not connected!')
    """
#    curr_positions.update(lxt_ttc_1.get_timing())
#    curr_positions.update(lxt_ttc_2.get_timing())
    curr_positions.update(lxt_ttc.ip1.get_timing())
    curr_positions.update(lxt_ttc.ip2.get_timing())
#    msg = msg if msg is not None else "I got timing!"
    kwargs['tags'] = kwargs.get('tags', ' ') + ' fs_timing'      
    _log_positions(curr_positions, log, msg=msg, **kwargs)


class PCavSig(MultiDerivedSignalRO):
    def _phase_to_ps(self, mds: MultiDerivedSignal, items: SignalToValue) -> float:
        return items[mds.signals[0]] * 1.06837606838
    calculate_on_get=_phase_to_ps
    
    def describe(self):
        return {
            self.name:{
                'source':'SIOC:UNDS:PT11:0:CAV0PRB1COMPPHASE',
                 'dtype':'number',
                 'shape':[], 
                 'units':'ps',
                 'precision':3
                 }
            }
    def describe_configuration(self):
        return self.describe()
    
class LaserTimingFeedback(BaseInterface, Device):
    """
    Class for controlling the timing feedback using the "ATM_FBK_OFFSET" PV. 
    The feedback can use either the phase cavity information or data from the shared memory for the atm edge position.
    The script uses a rolling median filter to filter the readings:  median += history * sign(sample -median), where the history is an adjust able parameter.      
    """    
    
    tab_component_names = True
    tab_whitelist = ['enable_feedback', 'disable_feedback','correction_sign','device', 'info', 'initial_offset']
    
    atm_fbk_offset = Cpt(EpicsSignal, ':PHASCTL:ATM_FBK_OFFSET', kind='normal')
    pcav_phase = FCpt(EpicsSignalRO, 'SIOC:UNDS:PT11:0:CAV0PRB1COMPPHASE', kind='normal')
    atm_diff = FCpt(EpicsSignal, 'TMO:VARS:TTZF_sPost1', kind='normal')
    #FCpt(EpicsSignalRO, 'TMO:TIMETOOL:TTALL', kind='normal')
#    pcav_ps=Cpt(MultiDerivedSignalRO, attrs=['pcav_phase'], calculate_on_get=_phase_to_ps)
    pcav_ps=Cpt(PCavSig, attrs=['pcav_phase']) 
    pcav_valid_cnt = FCpt(EpicsSignalRO, 'SIOC:UNDS:PT11:0:VALIDCNT0', kind='normal')
    shmem_status = FCpt(EpicsSignal, 'TMO:VARS:TTZF_sStatus', kind='normal')
    
    
    
    def __init__(self, prefix, **kwargs):
        #set some initial parameters
        self.device = 'pcav'
        self.correction_sign = 1
        self._initial_device_reading_ns = 0
        self._initial_fbk_offset = 0
        self._device_median_ns = 0
        self._pcav_conv = 1E9/(360*2.6E9) #1.06837606838E-3
        self.pcav_history = 5E-6 #history factor for median filter
        self._atm_conv = 5E-6 # atm is 5 fs/pix
        self.atm_history = 1E-6 #history factor for median filter
        return super().__init__(prefix, **kwargs) 
    
    def enable_feedback(self, device=None, initial_device_setting=None):
        """
        Start feedback using the device specified in the device arguement. 
        This script uses a rolling median filter to apply an offset to "ATM_FBK_OFFSET" PV.
        median += history * sign(sample -median), where the history is an adjust able parameter. 

        Parameters
        ----------
        device : str, optional
            Device to use for feedback. Should be 'pcav' or 'atm'. The default is 'pcav'.
        initial_device_setting : float, optional
            Define an initial offset for the feedback. The default is None, which will result in using the current reading of the feedback device.
            This is in units of the feedback device. 

        Returns
        -------
        None.

        """
        def _lock_cb(*args, value, timestamp, **kwargs):
            
            if abs(value) < 1E-6:
                #ignore zero values which are probably wrong. 
                return
            
            if self.device == 'pcav':
                conversion = self._pcav_conv
                history_f = self.pcav_history
            elif self.device =='atm':
                conversion = self._atm_conv
                history_f = self.atm_history
            
#            if self.pcav_valid_cnt.get() > 10: #If pcav is valid -> update median
                #rolling median filter
                # median += history * sign(sample -median)
            self._device_median_ns += (history_f)*np.sign(value*conversion - self._device_median_ns)
#            else:
#                logger.debug('pcav valid counts: %s' % self.pcav_valid_cnt.get() )
                
            current_diff_ns = (self._device_median_ns - self._initial_device_reading_ns) 
                
            current_offset_ns = np.sign(self.correction_sign) * current_diff_ns + self._initial_fbk_offset
            
            if debug_mode() is True:
                #debug mode writes to screen
                logger.debug('value: %s, median: %s, current diff ns: %s, current offset: %s' % (value, self._device_median_ns, current_diff_ns, current_offset_ns))
            else:
                #debug false writes to PV
                self.atm_fbk_offset.put( current_offset_ns ) #apply correct sign
            
            #add global stop
            if self.shmem_status.get() == 0:
                self.atm_diff.unsubscribe_all()
                self.pcav_phase.unsubscribe_all()
            return
        
        
        if device is None:
            self.device = 'pcav'
            logger.info('No device specified, defaulting to phase cavity.')     
        else:
            self.device = device
            logger.info('Using %s for feedback' % self.device)
            
        if self.shmem_status.get() != 0:
            logger.error('Feedback script is running elsewhere!')
            return
            
        #use the PV for non-debug to add offset globally.
        self._initial_fbk_offset=self.atm_fbk_offset.get()
        logger.info('Found an initial value of %s in the feedback PV' % self._initial_fbk_offset)
        
        
        if initial_device_setting is None:
            if self.device == 'pcav':
                #Find initial offset from the last hour of archive data.
                arch_data = archive.get(self.pcav_phase.pvname, xarray=True, 
                                        start=datetime.datetime.now()-datetime.timedelta(minutes=5), 
                                        end=datetime.datetime.now())
                filt_data = arch_data.where(arch_data != 0)
                self._initial_device_reading_ns = np.nanmedian(filt_data[self.pcav_phase.pvname][0]) * self._pcav_conv
                logger.debug('initial device reading ns: %s, initial reading: %s' % (self._initial_device_reading_ns, self._initial_device_reading_ns/self._pcav_conv))
            elif self.device == 'atm':
                #Initial offset is the current position.
                self._initial_device_reading_ns = self.atm_diff.get() * self._atm_conv
                
                #This probably needs to be improved.... this will get a stale value. 
        else:
            if self.device == 'pcav':
                self._initial_device_reading_ns =  initial_device_setting * self._pcav_conv
            elif self.device == 'atm':
                self._initial_device_reading_ns =  initial_device_setting * self._atm_conv
        
        self._device_median_ns = np.copy(self._initial_device_reading_ns)
               
        if self.device == 'pcav':
            self.shmem_status.put(1) #set shmem state 
            self._uid = self.pcav_phase.subscribe(_lock_cb) #start feedback
        
        elif self.device == 'atm':
            self.shmem_status.put(2) #set shmem state
            self._uid = self.atm_diff.subscribe(_lock_cb) #start feedback
        
        else:
            logger.error('feedback device not recognizied.')
            return
        
        sys.stdout.write("\x1b]2;lxt fbk\x07")
        return 
        
    def disable_feedback(self):
        status = self.shmem_status.get()
        if status == 1 or (status==0 and self.device=='pcav'):
            self.pcav_phase.unsubscribe_all()
            self.shmem_status.put(0)
        elif status == 2 or (status==0 and self.device=='atm'):
            self.atm_diff.unsubscribe_all()
            self.shmem_status.put(0)
            #add atm disable
        return
    
    def info(self):
        info_dict = OrderedDict()
        info_dict['lock status'] = self.shmem_status.get()
        info_dict['current correction'] = self.atm_fbk_offset.get()
        info_dict['current phase cavity [deg]'] = self.pcav_phase.get()
        info_dict['current atm diff [pix]'] = self.atm_diff.get()
        info_dict['lock device']= self.device
        info_dict['sign of correction'] = self.correction_sign
        info_dict['inital device reading [ns]'] = self._initial_device_reading_ns
        info_dict['initial fbk offset [ns]'] = self._initial_fbk_offset
        info_dict['pcav conversion'] = self._pcav_conv
        info_dict['pcav history'] = self.pcav_history
        info_dict['atm conversion'] = self._atm_conv
        info_dict['atm_history'] = self.atm_history
        _log_positions(info_dict, log=False)
        return
        
lxt_fbk = LaserTimingFeedback(TMO_LASER_TIME_PV, name='lxt_fbk')  

def _get_divisor(mds: MultiDerivedSignal, items: SignalToValue) -> int:
    if items[mds.signals[0]] > 0:
        #return round(items[mds.signals[1]]*0.98/items[mds.signals[0]])
        return round(items[mds.signals[1]]*0.97986/items[mds.signals[0]])
    else:
        return 0

def _get_overlap(mds: MultiDerivedSignal, items: SignalToValue) -> float:
    if items[mds.signals[0]] > 0:
        return ( np.lcm(items[mds.signals[0]],items[mds.signals[1]]) / items[mds.signals[0]] )
    else:
        return 0

def _get_tot_rate(mds: MultiDerivedSignal, items: SignalToValue) -> float:
    return items[mds.signals[0]] + items[mds.signals[1]]

#def _get_goose_delay(mds: MultiDerivedSignal, items: SignalToValue) -> float:
#    return ( (items[mds.signals[1]] - items[mds.signals[0]]) / (items[mds.signals[2]]*1E-3) )

class LaserTpr(BaseInterface, Device):
    tab_component_names= True
    tab_whitelist = ['summary']
    
    fid_rate = Cpt(EpicsSignalRO,':FIDRATE')
    sys_clk = Cpt(EpicsSignalRO, ':SYS2_CLK')
    carbide_base_tpr = Cpt(TprTrigger, '', channel=1, timing_mode=TimingMode.LCLS2)
    carbide_goose_tpr = Cpt(TprTrigger, '', channel=0, timing_mode=TimingMode.LCLS2)
    carbide_pp_tpr = Cpt(TprTrigger, '', channel=5, timing_mode=TimingMode.LCLS2)
    carbide_pp_goose_tpr = Cpt(TprTrigger, '', channel=4, timing_mode=TimingMode.LCLS2)
    laser_rate = Cpt(MultiDerivedSignalRO, attrs=['carbide_pp_tpr.eventrate','carbide_pp_goose_tpr.eventrate'], calculate_on_get=_get_tot_rate)
    laser_divisor = Cpt(MultiDerivedSignalRO, attrs=['laser_rate', 'fid_rate'], calculate_on_get=_get_divisor)
    goose_divisor = Cpt(MultiDerivedSignalRO, attrs=['carbide_pp_goose_tpr.eventrate','fid_rate'], calculate_on_get=_get_divisor)
    goose_freq_pulses = Cpt(MultiDerivedSignalRO, attrs=['laser_divisor', 'goose_divisor'], calculate_on_get=_get_overlap)
#    goose_delay = Cpt(MultiDerivedSignalRO, attrs=['carbide_pp_tpr.delay_ticks', 'goose_tpr.delay_ticks', 'sys_clk'], calculate_on_get=_get_goose_delay)
    
    def _get_goose_delay(self):
        op_pp = self.carbide_pp_tpr.operation.get()
        op_base = self.carbide_base_tpr.operation.get()
        
        if op_pp == 0:
            return 0
        elif op_base == 1: 
            return (self.carbide_pp_goose_tpr.delay_ticks.get() - self.carbide_pp_tpr.delay_ticks.get()) / (self.sys_clk.get()*1E-3)
        elif op_base == 2:
            return self.carbide_goose_tpr.width_ticks.get() / (self.sys_clk.get()*1E-3)
    
    def summary(self, log=False, **kwargs):
        """
        Get the settings of the laser timing TPR.
        
        Parameters:
        ---------------
        log - booleam to log the positions in the elog
        msg - optional argument to append a message to the elog with the positions.
        tag - tags to add to the elog post. 
        """
        timingDict = {'On-Time Rate:':self.carbide_pp_tpr.eventrate.get()/0.98,
                      'On-Time Sequence Code:': str(self.carbide_pp_tpr.seqcode.get()),
                      'Goose Rate:':self.carbide_pp_goose_tpr.eventrate.get()/0.98,
                      'Goose Sequence Code:': str(self.carbide_pp_goose_tpr.seqcode.get()),
                      'Goose Delay [ns]:':self._get_goose_delay(),#self.goose_delay.get(),
                      'Goose Overlap [pulses]:':self.goose_freq_pulses.get(),
                      'Goose Eanble:': bool(self.carbide_pp_tpr.operation.get())
                      }
        kwargs['tags'] = kwargs.get('tags', ' ') + ' laser_trigger'
        _log_positions(timingDict, log, **kwargs)
        
        
tmo_laser_tpr = LaserTpr(TMO_LASER_TPR_PV, name='laser_tpr')


def get_laser_tpr(log=False, **kwargs):
    """
    Get the settings of the laser timing TPR.
    
    Parameters:
    ---------------
    log - booleam to log the positions in the elog
    msg - optional argument to append a message to the elog with the positions.
    tag - tags to add to the elog post. 
    """
    tmo_laser_tpr.summary(log, **kwargs)
      