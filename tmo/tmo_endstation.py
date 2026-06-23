from tmo.db import tmo_cvmi, tmo_lamp, tmo_mrco, tmo_dream, elog
#from tmo.db import dream_gsjn, dream_gsjp, dream_coil
#from tmo.tmo_beamline_control import _log_positions
from tmo.tmo_utils import get_motion as _get_motion
from tmo.tmo_utils import log_positions as _log_positions
from tmo.db import camviewer
from ophyd.device import Component as Cpt
from ophyd.signal import EpicsSignal, EpicsSignalRO
from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from pcdsdevices.valve import VCN_VAT590
from pcdsdevices.rs_powersupply import RSChannel, RohdeSchwarzPowerSupply
import pandas as pd
import logging
logger = logging.getLogger(__name__)
# from pcdsdevices.sqr1 import SQR1
from tmo.sqr1 import SQR1
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import subprocess
from tmo.tmo_mpods import mpod_name_from_file
from tmo.tmo_dream_utils import tmo_dream_sample_paddle
tmo_lamp_sq1 = SQR1('MCQ:TMO:01', name='tmo_lamp_sq1')
tmo_cvmi_sq1 = SQR1('MCQ:TMO:02', name='tmo_cvmi_sq1')
tmo_mrco_sq1 = SQR1('MCQ:TMO:03', name='tmo_mrco_sq1')


def exp_switch(IP='ip1', end_station=None):
    """
    Change softlinks for the DAQ related variables and reload mpod names from file.  

    Parameters
    ----------
    IP : TYPE, optional
        DESCRIPTION. The default is 'ip1'.
    end_station : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    elog_msg = 'Experiment switch'
    logger.debug('IP is %s, endstation is %s' % (IP, end_station))
    if IP.lower() == 'ip1':
        msg = 'updated epicsArch_laser.txt -> epicsArch_laser_ip1.txt'
        subprocess.run(['ln', '-snf', 'epicsArch_laser_ip1.txt', '/cds/home/opr/tmoopr/misc/epicsArch_laser.txt'])
        logger.info(msg)
        elog_msg += ('\n' + msg)
        
        if end_station.lower() == 'mrco':
            msg = 'switching to mrco'
            elog_msg += ('\n' + msg)
            logger.debug(msg)
            
            msg = 'updated epicsArch_tmo_endstation.txt -> epicsArch_mrco_endstation.txt'
            subprocess.run(['ln', '-snf', 'epicsArch_mrco_endstation.txt', '/cds/home/opr/tmoopr/misc/epicsArch_tmo_endstation.txt'])
            elog_msg += ('\n' + msg)
            logger.info(msg)

            mpod_name_from_file(filename='/cds/home/opr/tmoopr/hutch-python/tmo/tmo_mpod_configs/mrco.json')
            elog_msg += ('\n' + 'update mpod names to mrco.json')
            
            msg = 'updated tmo.py -> tmo_sc_mrco.py'
            subprocess.run(['ln', '-snf', 'tmo_sc_mrco.py', '/cds/home/opr/tmoopr/daq/tmo.py'])
            elog_msg += ('\n' + msg)
            logger.info(msg)
        
        elif end_station.lower() == 'mbes': 
            msg = 'switching to mbes'
            elog_msg += ('\n' + msg)
            logger.debug(msg)
            
            msg = 'updated epicsArch_tmo_endstation.txt -> epicsArch_mbes.txt'
            subprocess.run(['ln', '-snf', 'epicsArch_mbes.txt', '/cds/home/opr/tmoopr/misc/epicsArch_tmo_endstation.txt'])
            elog_msg += ('\n' + msg)
            logger.info(msg)
            
            mpod_name_from_file(filename='/cds/home/opr/tmoopr/hutch-python/tmo/tmo_mpod_configs/mbes.json')
            elog_msg += ('\n' + 'update mpod names to mbes.json')
            
            msg = 'updated tmo.py -> tmo_sc_mbes.py'
            subprocess.run(['ln', '-snf', 'tmo_sc_mbes.py', '/cds/home/opr/tmoopr/daq/tmo.py'])
            elog_msg += ('\n' + msg)
            logger.info(msg)
        
        else:
            logger.error('end_station not regognizied.')
        
        msg = 'switch to A0_2 inputs for HSD'
        logger.info(msg)
        subprocess.run(['source ~/daq/setup_env.sh && hsd_input_select -H TMO -I 0'], shell=True, executable="/bin/bash")
        elog_msg += ('\n' + msg)
    
    elif IP.lower() == 'ip2':
        msg = 'updated epicsArch_laser.txt -> epicsArch_laser_ip2.txt'
        subprocess.run(['ln', '-snf', 'epicsArch_laser_ip2.txt', '/cds/home/opr/tmoopr/misc/epicsArch_laser.txt'])
        elog_msg += ('\n' + msg)
        logger.info(msg)
        
        msg = 'updated epicsArch_tmo_endstation.txt -> epicsArch_dream_endstation.txt'
        subprocess.run(['ln', '-snf', 'epicsArch_dream_endstation.txt', '/cds/home/opr/tmoopr/misc/epicsArch_tmo_endstation.txt'])
        elog_msg += ('\n' + msg)
        logger.info(msg)
        
        mpod_name_from_file(filename='/cds/home/opr/tmoopr/hutch-python/tmo/tmo_mpod_configs/dream.json')
        elog_msg += ('\n' + 'update mpod names to dream.json')
        
        msg = 'updated tmo.py -> tmo_sc_dream.py'
        subprocess.run(['ln', '-snf', 'tmo_sc_dream.py', '/cds/home/opr/tmoopr/daq/tmo.py'])
        elog_msg += ('\n' + msg)
        logger.info(msg)
        
        msg = 'switch to A1_3 inputs for HSD'
        logger.info(msg)
        subprocess.run(['source ~/daq/setup_env.sh && hsd_input_select -H TMO -I 1'], shell=True, executable="/bin/bash")
        elog_msg += ('\n' + msg)
        
    else:
        logger.error('IP not recognizied.')    
    
    elog.post(elog_msg, facility=True, experiment=False)

def  get_lamp_sq1(log=False, msg=None, update_preset=None, preset_type='exp', **kwargs):
    """
    Get current position of the LAMP sq1 stand.
    
    Parameters
    ----------
    log : bool, optional
        Flag to post the status to elog. The default is False.
    msg : str, optional
        Message to include in elog post. The default is None.
    run : int, optional
        Associate the post with a specific run
    tags : list, optional
        List of tags to add to the post
    
    """
    def _update_presets(update_preset, msg):
        axis_list = [tmo_lamp_sq1.x,tmo_lamp_sq1.y,tmo_lamp_sq1.z, tmo_lamp_sq1.rx, tmo_lamp_sq1.ry, tmo_lamp_sq1.rz]
        for axis in axis_list:
            if preset_type == 'exp':
                axis.presets.add_here_exp(name=update_preset, comment=msg)
            elif preset_type == 'hutch':
                axis.presets.add_here_hutch(name=update_preset, comment=msg)
            else:
                logger.error('preset_type not recognizied. Should be "exp" or "hutch".')
                return None
        msg = 'Update ' + update_preset + ' for '+preset_type+' with current axis positions. \n'
        return msg
    
    curr_positions = {}
    for keys in dir(tmo_lamp_sq1):
        motor = getattr(tmo_lamp_sq1, keys)
        if hasattr(motor, 'position'):
            curr_positions[keys] = getattr(motor, 'position')
    kwargs['tags'] = kwargs.get('tags', ' ') + ' sq1'        
    
    if update_preset is not None:
        update_msg = _update_presets(update_preset, msg)
        if update_msg is not None:
            log = True
            kwargs['tags'] = kwargs.get('tags', ' ') + ' presets'
        msg = update_msg + (msg or '')
        
    _log_positions(curr_positions, log, msg, **kwargs)

def set_lamp_sq1(sq1_preset=None):
    """
    Move the LAMP SQ1 to a preset position. Passing no position will return the list of existing presets.

    Parameters
    ----------
    sq1_preset : str, optional
        name of preset position. The default is None.
    """
    axis_list = [tmo_lamp_sq1.x, tmo_lamp_sq1.y, tmo_lamp_sq1.z]
    if sq1_preset is not None:
        if any(hasattr(a.presets.positions, sq1_preset) for a in axis_list):
            get_lamp_sq1(log=True, msg='SQ1 position before moving to preset position ' +sq1_preset)
            x_sp = getattr(tmo_lamp_sq1.x.presets.positions, sq1_preset)
            y_sp = getattr(tmo_lamp_sq1.y.presets.positions, sq1_preset)
            z_sp = getattr(tmo_lamp_sq1.z.presets.positions, sq1_preset)
            tmo_lamp_sq1.multi_axis_move(x_sp=x_sp, y_sp=y_sp, z_sp=z_sp) 
            get_lamp_sq1(log=True, msg=('Move KB1 to preset position '+sq1_preset))
        else:
            logger.error('One of the axes is missing the desired state.')
    else:         
        state_dict={"State:": []}       
        for axis in axis_list:
            state_dict[axis.name] = []
            for state in list(vars(axis.presets.positions).keys()):
                if state not in state_dict['State:']:
                    state_dict['State:'].append(state)
        for state in state_dict['State:']:
            for axis in axis_list:
                state_dict[axis.name].append(vars(axis.presets.positions).get(state, ' '))
        print (pd.DataFrame(state_dict))
    
def  get_cvmi_sq1(log=False, msg=None, **kwargs):
    """
    Get current position of the LAMP sq1 stand.
    
    Parameters
    ----------
    log : boll, optional
        Flag to post the status to elog. The default is False.
    msg : str, optional
        Message to include in elog post. The default is None.
    include_cam : list of str, optional
        list of camera names to append to elog post.
    run : int, optional
        Associate the post with a specific run
    tags : list, optional
        List of tags to add to the post
    
    """
    
    curr_positions = {}
    for keys in dir(tmo_cvmi_sq1):
        motor = getattr(tmo_cvmi_sq1, keys)
        if hasattr(motor, 'position'):
            curr_positions[keys] = getattr(motor, 'position')
    kwargs['tags'] = kwargs.get('tags', ' ') + ' sq1'        
    _log_positions(curr_positions, log, msg, **kwargs)

def  get_mrco_sq1(log=False, msg=None, **kwargs):
    """
    Get current position of the LAMP sq1 stand.
    
    Parameters
    ----------
    log : boll, optional
        Flag to post the status to elog. The default is False.
    msg : str, optional
        Message to include in elog post. The default is None.
    include_cam : list of str, optional
        list of camera names to append to elog post.        
    run : int, optional
        Associate the post with a specific run
    tags : list, optional
        List of tags to add to the post
    
    """
    
    curr_positions = OrderedDict()
    for keys in dir(tmo_mrco_sq1):
        motor = getattr(tmo_mrco_sq1, keys)
        if hasattr(motor, 'position'):
            curr_positions[keys] = getattr(motor, 'position')
    kwargs['tags'] = kwargs.get('tags', ' ') + ' sq1'        
    _log_positions(curr_positions, log, msg, **kwargs)
    
    
def get_cvmi_motion(log=False, msg=None, include_cam=None, **kwargs):
    """
    Get current position of the cVMI motion.   
    """
    
    devices = [tmo_cvmi, tmo_cvmi_sq1]
    kwargs['tags'] = kwargs.get('tags', ' ') + ' cvmi_positions'        
    _get_motion(devices, log, msg, include_cam, **kwargs)

get_cvmi_motion.__doc__ += _log_positions.__doc__

def get_lamp_motion(log=False, msg=None, include_cam=None, **kwargs):
    """
    Get current position of the LAMP motion.    
    """
        
    devices = [tmo_lamp, tmo_lamp_sq1]
    kwargs['tags'] = kwargs.get('tags', ' ') + ' lamp_positions'        
    _get_motion(devices, log, msg, include_cam, **kwargs)

get_lamp_motion.__doc__ += _log_positions.__doc__

def get_mrco_motion(log=False, msg=None, include_cam=None, **kwargs):
    """
    Get current position of the MRCO motion.  
    """
    
    devices = [tmo_mrco, tmo_mrco_sq1]
    kwargs['tags'] = kwargs.get('tags', ' ') + ' mrco_positions'        
    _get_motion(devices, log, msg, include_cam, **kwargs) 

get_mrco_motion.__doc__ += _log_positions.__doc__

def get_dream_motion(log=False, msg=None, include_cam=None, **kwargs):
    """
    Get current position of the DREAM motion.    
    """
    
#    devices = [dream_gsjn, dream_gsjp, dream_coil]
    devices = [tmo_dream, tmo_dream_sample_paddle]
    kwargs['tags'] = kwargs.get('tags', ' ') + ' dream_positions'        
    _get_motion(devices, log, msg, include_cam, **kwargs)

get_dream_motion.__doc__ += _log_positions.__doc__
    
def mv_to_preset(device, preset):
    """
    Move the 'device' to a preset position described by 'preset'.
    Parameters
    ----------
    device : object with multiple axes
        Device with multiple axes which you want to move to a preset location
    preset : str.
        Name of preset position.

    Returns
    -------
    None.

    """
    for keys in dir(device):
        motor = getattr(device, keys)
        if hasattr(motor, 'presets'):
            if (motor.presets.has_presets):
                if (preset in motor.presets.positions.__dict__):
                    move_to_preset = getattr(motor, 'mv_'+preset )
                    move_to_preset

class HutchVCN(BaseInterface, Device):
    """
    Class for remote controled leak valve. 
    """
    tab_component_names = True
    tab_whitelist = ['open_valve', 'close_valve','set_manual_cntl','set_pres_cntl','status']


    valve = Cpt(VCN_VAT590, '' )
    
    def open_valve(self):
        """
        Open the valve to the opening limit
        """
        self.valve.state.put(2)
        
    def close_valve(self):
        """
        Close the valve
        """
        self.valve.state.put(1)
        
    def set_manual_cntl(self, set_point=0):
        """        
        Set the valve to manual control and open to set_point

        Parameters
        ----------
        set_point : float, optional
            Percentage to open the valve. The default is 0.

        """
        self.valve.pos_req.put(0)
        self.valve.state.put(0)
        self.valve.pos_req.put(set_point)
        
    def set_pres_cntl(self, set_point=0):
        """
        Set the valve to pressure control and set the control pressure to set_point

        Parameters
        ----------
        set_point : float, optional
            Pressure setpoint in Torr. The default is 0.

        """
        self.valve.pres_req.put(0)
        self.valve.state.put(3)
        self.valve.pres_req.put(set_point)
        
    def status(self, log=False, msg=None, **kwargs):
        """
        Print the status of the valve

        Parameters
        ----------
        log : boll, optional
            Flag to post the status to elog. The default is False.
        msg : str, optional
            Message to include in elog post. The default is None.
        run : int, optional
            Associate the post with a specific run
        tags : list, optional
            List of tags to add to the post

        """
        valve_states = ['ManualControl', 'CloseValve', 'OpenValve', 'PressureControl', 'HoldValve']
        status_dict={
            'control mode': valve_states[self.valve.state.get()],
            'pressure setting (10-8 torr)': self.valve.pres_req.get()*1E8,
            'pressure reading (10-8 torr)': self.valve.pres_torr.get()*1E8,
            'position setting': self.valve.pos_raw.get()
            }
        _log_positions(status_dict, log, msg, **kwargs)

tmo_lamp_vcn = HutchVCN('LAMP:VCN:01:',name='tmo_lamp_vcn')
tmo_mrco_vcn = HutchVCN('MRCO:VCN:01:', name='tmo_mrco_vcn')

def get_lamp_vcn(log=False, msg=None, **kwargs):
        """
        Print the status of the remote controlled VAT-590 valve on LAMP

        """
        tmo_lamp_vcn.status(log, msg, **kwargs)
get_lamp_vcn.__doc__ += _log_positions.__doc__

def get_mrco_vcn(log=False, msg=None, **kwargs):
        """
        Print the status of the remote controlled VAT-590 valve on MRCO

        """
        tmo_mrco_vcn.status(log, msg, **kwargs)
get_mrco_vcn.__doc__ += _log_positions.__doc__

class BKP(BaseInterface, Device):
    
    tab_component_names= True
    tab_whitelist = []
    
    voltage = Cpt(EpicsSignal, ':AVDD_VOLT', write_pv=':SETVOLTAGE_RAW_IN', kind='normal')
    current = Cpt(EpicsSignal, ':AVDD_CUR', write_pv=':SETCURRENT_RAW_IN', kind='normal')
    volttage_limit = Cpt(EpicsSignal, ':LIMIT_V', write_pv=':SETUPLIMIT_IN', kind='omitted')
    mode = Cpt(EpicsSignalRO, ':MODE', kind='normal')
    max_voltage = Cpt(EpicsSignalRO,':MAX_V', kind='normal')
    max_current = Cpt(EpicsSignalRO, ':MAX_I', kind='normal')
    _set_on = Cpt(EpicsSignal, ':SETAVDDON', kind='omitted')
    _set_off = Cpt(EpicsSignal, ':SETAVDDOFF', kind='omitted')
    desc = Cpt(EpicsSignal, ':SETVOLTAGE_RAW_IN.DESC', string=True, kind='normal')
    
    def on(self):
        self._set_on.put(1)
        
    def off(self):
        self._set_off.put(1)
 
class FPS(BaseInterface, Device):
    
    tab_component_names= True
    tab_whitelist = ['on', 'off']
    
    voltage = Cpt(EpicsSignal, ':GetVoltage',write_pv=':SetVoltage', kind='normal')
    volt_ramp = Cpt (EpicsSignal, ':GetVoltageRampSpeed', write_pv=':SetVoltageRampSpeed', kind='normal')
    _voltage_set = Cpt(EpicsSignalRO, ':GetVoltagaeSetvalue')
    current = Cpt(EpicsSignal, ':GetCurrent', write_pv=':SetCurrent', kind='normal')
    curr_ramp = Cpt(EpicsSignal, ':GetCurrentRampSpeed', write_pv=':SetCurrentRampSpeed', kind='normal')
    _current_set = Cpt(EpicsSignalRO, ':GetCurrentSetvalue')
    current_limit = Cpt(EpicsSignal, ':GetCurrentLimitSetvalue', write_pv = ':SetCurrentLimit', kind='normal')
    voltage_limit = Cpt(EpicsSignal, ':GetVoltageLimitSetvalue', write_pv = ':SetVoltageLimit', kind='normal')
    status = Cpt(EpicsSignal, ':GetStatus')
    _on_switch = Cpt(EpicsSignal, ':SetSwitchONOFF', kind='omitted')
    
    def on(self):
        self._on_switch.put(1)
        
    def off(self):
        self._on_switch.put(0)

# class RSChannel(RSChannel):
#     desc = Cpt(EpicsSignal, ':SET_VOLT.DESC', kind='normal')
    
# class RohdeSchwarzPowerSupply(RohdeSchwarzPowerSupply):
        
#     tab_component_names= True
#     tab_whitelist = ['on', 'off']

# class TmoRS(BaseInterface, Device):
#     tab_component_names= True  

#     rs_1 = Cpt(RohdeSchwarzPowerSupply)
#     rs_2 =          

class TmoBkp(BaseInterface, Device):
    tab_component_names= True
    tab_whitelist = []
    
    bk01 = Cpt(BKP, ':01')
    bk02 = Cpt(BKP, ':02')
    bk03 = Cpt(BKP, ':03')
    bk04 = Cpt(BKP, ':04')

tmo_bkp = TmoBkp('TMO:BKP', name='tmo_bkp')

class TmoFps(BaseInterface, Device): 
        tab_component_names= True
        tab_whitelist = []
        
        fps01 = Cpt(FPS, ':01')
        fps02 = Cpt(FPS, ':02')
        fps03 = Cpt(FPS, ':03')       

tmo_fps = TmoFps('TMO:FPS', name='tmo_fps')
