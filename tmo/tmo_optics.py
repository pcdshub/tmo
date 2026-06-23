from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from ophyd.signal import EpicsSignal, EpicsSignalRO
from ophyd.status import DeviceStatus
from pcdsdevices.device import Component as Cpt
from pcdsdevices.device import ObjectComponent as OCpt
from ophyd import FormattedComponent as FCpt
from pcdsdevices.epics_motor import BeckhoffAxisNoOffset
from pcdsdevices.pseudopos import SyncAxis
from pcdsdevices.pv_positioner import OnePVMotor
from pcdsdevices.tpr import TprTrigger, TimingMode
from pcdsdevices.slits import SL2K4Slits
from tmo.db import mr2k4_kbo, mr3k4_kbo, mr1k4_soms, mr4k4_kbo, mr5k4_kbo
from tmo.db import sl1k0, sl2k0, sl1k4, im5k4, pf1k4, sp1k4
from tmo.db import camviewer, archive
from tmo.tmo_beamline_control import at1k0
from tmo.tmo_utils import log_positions as _log_positions
from tmo.tmo_utils import motor_arch_compare, SyncAxisTmo
from collections import OrderedDict
import pandas as pd
from tabulate import tabulate
import numpy as np
import datetime
import time
import logging
logger = logging.getLogger(__name__)

sl2k4 = SL2K4Slits(prefix="SL2K4:SCATTER", name="tmo_sl2k4",cam="IM6K4:PPM:CAM", data_source="IMAGE2")

def get_KB1(log=False, msg=None, save_bender_preset=None, preset_type='hutch', **kwargs):
    """
    Get the current position of the first KB pitch, bender, and motion. This can be saved to the elog. The positions can also be saved as presets to be used later.

    Parameters
    ----------
    log : bool, optional
        boolean to log the positions in the elog. The default is False.
    msg : str, optional
        Optional argument to append a message to the elog post. The default is None.
    tag : str, optional
        List of tags to add to an elog post, by default kbo1 is tagged
    save_bender_preset : str, optional
        The name of the preset position to create or update. The default is None, which creates no preset
    preset_type : str, optional
        Str to specify if the preset is associated with the 'hutch' or the current 'exp'eriment. The default is 'hutch'.
    """
    def _update_presets(save_bender_preset, msg):
        bender_list = [mr2k4_kbo.bender_us, mr2k4_kbo.bender_ds, mr3k4_kbo.bender_us, mr3k4_kbo.bender_ds]
        for bender in bender_list:
            bender.presets.sync()
            if preset_type == 'exp':
                bender.presets.add_here_exp(name=save_bender_preset, comment=msg)
            elif preset_type == 'hutch':
                bender.presets.add_here_hutch(name=save_bender_preset, comment=msg)
            else:
                logger.error('preset_type not recognizied. Should be "exp" or "hutch".')
                return None
        msg = 'Update ' + save_bender_preset + ' for '+preset_type+' with current bender positions. \n'
        return msg
            
    if save_bender_preset is not None:
        update_msg = _update_presets(save_bender_preset, msg)
        if update_msg is not None:
            log = True
            kwargs['tags'] = kwargs.get('tags', ' ') + ' presets'
        if preset_type == 'hutch':
            kwargs['facility'] = True
        msg = update_msg + (msg or '')
    
    curr_positions= {}
    for keys in dir(mr2k4_kbo):
        motor = getattr(mr2k4_kbo, keys)
        if hasattr(motor, "position"):
            curr_positions["mr2k4 "+keys] = getattr(motor, 'position')
    
    curr_positions['mr2k4 bender relative'] = mr2k4_kbo_bender.rbk.position
    
    if mr2k4_kbo.bender_ds.presets.sync_needed(): mr2k4_kbo.bender_ds.presets.sync()
    bndr_preset = mr2k4_kbo.bender_ds.presets.state()
    if not (bndr_preset == 'Unknown'):
        curr_positions["mr2k4 bender preset"] = bndr_preset
    
    for keys in dir(mr3k4_kbo):
        motor = getattr(mr3k4_kbo, keys)
        if hasattr(motor, "position"):
            curr_positions["mr3k4 "+keys] = getattr(motor, 'position')
    
    curr_positions['mr3k4 bender relative'] = mr3k4_kbo_bender.rbk.position
    
    if mr3k4_kbo.bender_ds.presets.sync_needed(): mr3k4_kbo.bender_ds.presets.sync()
    bndr_preset = mr3k4_kbo.bender_ds.presets.state()
    if not (bndr_preset == 'Unknown'):
        curr_positions["mr3k4 bender preset"] = bndr_preset
    
    kwargs['tags'] = 'kbo1 ' + kwargs.get('tags', ' ')  

    _log_positions(curr_positions, log, msg, **kwargs)        

def set_KB1(bender_preset=None):
    """
    Move the KB1 benders to a preset position. Passing no position will return the list of existing presets.

    Parameters
    ----------
    bender_preset : str, optional
        name of preset position. The default is None.
    """
    bender_list = [mr2k4_kbo.bender_us, mr2k4_kbo.bender_ds, mr3k4_kbo.bender_us, mr3k4_kbo.bender_ds]
    if bender_preset is not None:
        for bender in bender_list: bender.presets.sync()
        if any(hasattr(b.presets.positions, bender_preset) for b in bender_list):
            get_KB1(log=True, msg='KB1 position before moving to preset position ' +bender_preset)
            for bender in bender_list:
                umv = getattr(bender, 'umv_'+bender_preset)
                umv()
            moving_status = True
            while moving_status:
                moving_status = False
                for bender in bender_list:
                    moving_status = moving_status or bender.moving
            
            mr2k4_kbo_bender.set_current_position(0)
            mr3k4_kbo_bender.set_current_position(0)
            
            get_KB1(log=True, msg=('Move KB1 to preset position '+bender_preset))
            
        else:
            logger.error('One of the motors is missing the desired state.')
    else:
        state_dict={"State:": []}       
        for bender in bender_list:
            bender.presets.sync()
            state_dict[bender.name] = []
            for state in list(vars(bender.presets.positions).keys()):
                if state not in state_dict['State:']:
                    state_dict['State:'].append(state)
        for state in state_dict['State:']:
            for bender in bender_list:
                state_dict[bender.name].append(vars(bender.presets.positions).get(state, ' ')) 
        print (pd.DataFrame(state_dict))

def restore_KB1(tstamp):
    """
    Restore IP1 KB mirrors back to a previous setting from the archiver. 
    After finding the archive data, the function will ask you to verify you want to restore. 

    Parameters
    ----------
    tstamp : datetime object or list
        date/time to restore from archiver data.

    Returns
    -------
    None.

    """
    if type(tstamp) is list:
        tstamp = datetime.datetime(*tstamp)
    elif not (type(tstamp) is datetime.datetime):
        logger.error('Arguement is not a valid time stammp')
        return
    
    curr_pos=OrderedDict()
    prev_pos=OrderedDict()
    axis_dict=OrderedDict()
    for mr in [mr2k4_kbo,mr3k4_kbo]:
        for keys in dir(mr):
            axis = getattr(mr, keys)
            if hasattr(axis, 'user_readback'):
                curr_pos[axis.name]=axis.position
                axis_dict[axis.name]=axis
                pv=axis.user_readback.pvname
                arch_data = archive.get(pv, xarray=True, start=tstamp, end=tstamp+datetime.timedelta(minutes=1))
                prev_pos[axis.name] = arch_data.sel(time=tstamp, method='nearest').to_dataframe()[pv]['vals']
       
    settings = {'Axis':curr_pos.keys(), 
                'Current':curr_pos.values(), 
                'Archive':prev_pos.values(), 
                'Difference':np.array(list(curr_pos.values()))-np.array(list(prev_pos.values()))}            
    settingsDF_str = tabulate(pd.DataFrame(settings), headers="keys", tablefmt="pretty")
    #print(pd.DataFrame(settings))
    print(settingsDF_str)
    
    response = input('\nWould you like to restore the previous condition?')
    if response.lower() == 'y' or response.lower() == 'yes':
    
        get_KB1(log=True, msg='KB1 position before restoring to previous condition.')
        for axis_name, axis in axis_dict.items():
            axis.mv(prev_pos[axis_name])
        moving = True
        while moving:
            time.sleep(2)
            moving = False
            for axis_name, axis in axis_dict.items():
                moving = moving or axis.moving
        get_KB1(log=True, msg='KB1 after restoring to previous condition.')
            
                    
        
def get_KB2(log=False, msg=None, **kwargs):
    """
    Get the current position of the second KB pitch, and motion. 

    Parameters
    ----------
    log : bool, optional
        boolean to log the positions in the elog. The default is False.
    msg : str, optional
        Optional argument to appena a message to the elog post. The default is None.
    tag : str, optional
        List of tags to add to an elog post, by default kbo1 is tagged
    """
    curr_positions= {}
    for keys in dir(mr4k4_kbo):
        motor = getattr(mr4k4_kbo, keys)
        if hasattr(motor, "position"):
            curr_positions["mr4k4 "+keys] = getattr(motor, 'position')
    for keys in dir(mr5k4_kbo):
        motor = getattr(mr5k4_kbo, keys)
        if hasattr(motor, "position"):
            curr_positions["mr5k4 "+keys] = getattr(motor, 'position')
    kwargs['tags'] = kwargs.get('tags', ' ') + ' kbo2'        
    _log_positions(curr_positions, log, msg, **kwargs)

def get_mr1k4(log=False, msg=None, **kwargs):
    """
    Get the current position of MR1K4 pitch, bender, and motion. 
    
    Parameters
    ----------
    log : bool, optional
        boolean to log the positions in the elog. The default is False.
    msg : str, optional
        Optional argument to appena a message to the elog post. The default is None.
    tag : str, optional
        List of tags to add to an elog post, by default kbo1 is tagged
    """
    curr_positions= {}
    for keys in dir(mr1k4_soms):
        motor = getattr(mr1k4_soms, keys)
        if hasattr(motor, 'connected'):
            if motor.connected:
                if hasattr(motor, "position"):
                    curr_positions["mr1k4 "+keys] = getattr(motor, 'position')
    kwargs['tags'] = kwargs.get('tags', ' ') + ' mr1k4'        
    _log_positions(curr_positions, log, msg, **kwargs)

def restore_mr1k4(tstamp):
    """
    Restore MR1K4 mirror back to a previous setting from the archiver. 
    After finding the archive data, the function will ask you to verify you want to restore. 

    Parameters
    ----------
    tstamp : datetime object or list
        date/time to restore from archiver data.

    Returns
    -------
    None.

    """
    
    if type(tstamp) is list:
        tstamp = datetime.datetime(*tstamp)
    elif not (type(tstamp) is datetime.datetime):
        logger.error('Arguement is not a valid time stammp')
        return
    
    curr_pos=OrderedDict()
    prev_pos=OrderedDict()
    axis_dict=OrderedDict()
    for keys in dir(mr1k4_soms):
        axis = getattr(mr1k4_soms, keys)
        if hasattr(axis, 'user_readback'):
            if axis.connected:
                curr_pos[axis.name]=axis.position
                axis_dict[axis.name]=axis
                pv=axis.user_readback.pvname
                arch_data = archive.get(pv, xarray=True, start=tstamp, end=tstamp+datetime.timedelta(minutes=1))
                prev_pos[axis.name] = arch_data.sel(time=tstamp, method='nearest').to_dataframe()[pv]['vals']
       
    settings = {'Axis':curr_pos.keys(), 
                'Current':curr_pos.values(), 
                'Archive':prev_pos.values(), 
                'Difference':np.array(list(curr_pos.values()))-np.array(list(prev_pos.values()))}            
    
    settingsDF_str = tabulate(pd.DataFrame(settings), headers="keys", tablefmt="pretty")
    #print(pd.DataFrame(settings))
    print(settingsDF_str)
    response = input('\nWould you like to restore the previous condition? ')
    if response.lower() == 'y' or response.lower() == 'yes':
    
        get_mr1k4(log=True, msg='MR1K4 position before restoring to previous condition.')
        for axis_name, axis in axis_dict.items():
            axis.mv(prev_pos[axis_name])
        moving = True
        while moving:
            moving = False
            for axis_name, axis in axis_dict.items():
                moving = moving or axis.moving
        get_mr1k4(log=True, msg='MR1K4 after restoring to previous condition.')
    
class CombinedBender(BaseInterface, Device):
    """
    Combined motion of upstrean and downstream benders to displace the focus.
    
    Parameters
    ----------
    step_size : list of floats, optional
        bender motion per mm. The default for mr2k4 is [0.02470623, 0.03143349] and mr3k4 is is [0.02569920, 0.03999258].
    """
    tab_whitelist = ['mvr', 'set_current_position', 'position']
    
    us = Cpt(BeckhoffAxisNoOffset,':MMS:BEND:US', kind='omitted')
    ds = Cpt(BeckhoffAxisNoOffset,':MMS:BEND:DS', kind='omitted')
    readback = Cpt(EpicsSignal, ':BEND:CMBD', kind='omitted')
    egu='mm'
    
    def __init__(self, *args, step_size=[0,0], **kwargs):
        self._step_size = step_size
        return super().__init__(*args, **kwargs)
    
    @property
    def position(self):
        return self.readback.get()
    
    def mvr(self, delta, **kwargs):
        """
        Relative move of mirror benders to move the focus a specified distance in mm.

        Parameters
        ----------
        step : float
        step size in mm.

        """
        #check limits before moving
        us_pos = self.us.wm()+delta*self._step_size[0]
        ds_pos = self.ds.wm()+delta*self._step_size[1]
        self.us.user_setpoint.check_value(us_pos)
        self.ds.user_setpoint.check_value(ds_pos)
#        us_move_ok = (us_pos > min(self.us.limits)) and (us_pos < max(self.us.limits)) 
#        ds_move_ok = (ds_pos > min(self.ds.limits)) and (ds_pos < max(self.ds.limits)) 
        
#        if us_move_ok and ds_move_ok:
        self.us.mvr(delta*self._step_size[0], **kwargs)
        self.ds.mvr(delta*self._step_size[1], **kwargs)
        self.readback.put(self.position + delta)            
#        if not us_move_ok:
#            logger.error('Requested motion outside of allowed range for upstream bender.')
#        if not ds_move_ok:
#            logger.error('Requested motion outside of allowed range for downstream bender.')
            
    def set_current_position(self, position):
        self.readback.put(position)
    
    def set(self, position):
        return
    
        
mr2k4_kbo_bender_old = CombinedBender('MR2K4:KBO', step_size=[0.02470623, 0.03143349], name='mr2k4_kbo_bender_comb')
mr3k4_kbo_bender_old = CombinedBender('MR3K4:KBO', step_size=[0.02569920, 0.03999258], name='mr3k4_kbo_bender_comb')

class Mr2k4Bender(SyncAxisTmo):
 
    us = OCpt(mr2k4_kbo.bender_us)
    ds = OCpt(mr2k4_kbo.bender_ds)
    rbk = Cpt(OnePVMotor, 'MR2K4:KBO:BEND:CMBD')
    
    tab_component_names = True
#    pos = sync * scale + offset
    scales = {'us': 0.02470623, 'ds': 0.03143349, 'rbk': 1}
    offsets = {'us': 13.378549,'ds': 19.577547, 'rbk': 0}
    
    warn_deadband = 0.01
    fix_sync_keep_still = 'us'
    sync_limits = (-1000, 1000)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_offset_to_current_position(self.rbk.position)
        
    def set_offset_to_current_position(self, fix_sync_pos=None):
        fix_sync_pos = fix_sync_pos if fix_sync_pos is not None else self.sync.position
        self.rbk.mv(fix_sync_pos, log=False)
        super().set_offset_to_current_position(fix_sync_pos)
        
    def set_current_position(self, position):
        self.set_offset_to_current_position(fix_sync_pos=position)
        
        
class Mr3k4Bender(SyncAxisTmo):
 
    us = OCpt(mr3k4_kbo.bender_us)
    ds = OCpt(mr3k4_kbo.bender_ds)
    rbk = Cpt(OnePVMotor, 'MR3K4:KBO:BEND:CMBD')

    tab_component_names = True
#    pos = sync * scale + offset
    scales = {'us': 0.02569920, 'ds': 0.03999258, 'rbk': 1}
    offsets = {'us': 17.601027,'ds': 17.656663, 'rbk': 0}
    
    warn_deadband = 0.01
    fix_sync_keep_still = 'us'
    sync_limits = (-1000, 1000)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_offset_to_current_position(self.rbk.position)
        
    def set_offset_to_current_position(self, fix_sync_pos=None):
        fix_sync_pos = fix_sync_pos if fix_sync_pos is not None else self.sync.position
        self.rbk.mv(fix_sync_pos, log=False)
        super().set_offset_to_current_position(fix_sync_pos)
        
    def set_current_position(self, position):
        self.set_offset_to_current_position(fix_sync_pos=position)
           
mr2k4_kbo_bender = Mr2k4Bender(name='mr2k4_kbo_bender_comb')
mr3k4_kbo_bender = Mr3k4Bender(name='mr3k4_kbo_bender_comb')

# def mr2k4_kbo_bender_mvr(step, step_size=[0.02470623, 0.03143349]):
#     """
#     Move mr2k4 benders to move the focus a specified amount/

#     Parameters
#     ----------
#     step : float
#         step size in mm.
#     step_size : list of floats, optional
#         bender motion per mm. The default is [0.02470623, 0.03143349].

#     """
#     mr2k4_kbo.bender_us.mvr(step*step_size[0])
#     mr2k4_kbo.bender_ds.mvr(step*step_size[1],wait=True)
    
# def mr3k4_kbo_bender_mvr(step, step_size=[0.02569920, 0.03999258]):
#     """
#     Move mr2k4 benders to move the focus a specified amount/

#     Parameters
#     ----------
#     step : float
#         step size in mm.
#     step_size : list of floats, optional
#         bender motion per mm. The default is [0.02569920, 0.03999258].

#     """
#     mr3k4_kbo.bender_us.mvr(step*step_size[0])
#     mr3k4_kbo.bender_ds.mvr(step*step_size[1],wait=True)
        
def mirror_arch_compare(tstamp):
    """
    Compare the current mirror positions to a previous condition from the archived data.  

    Parameters
    ----------
    tstamp : datetime object or list
        date/time for date to retrieve from archiver.

    Returns
    -------
    None.

    """

#    motor_arch_compare([mr1k4_soms, mr2k4_kbo,mr3k4_kbo, mr4k4_kbo, mr5k4_kbo], tstamp)

    if type(tstamp) is list:
        tstamp = datetime.datetime(*tstamp)
    elif not (type(tstamp) is datetime.datetime):
        logger.error('Arguement is not a valid time stammp')
        return
    
    curr_pos=OrderedDict()
    prev_pos=OrderedDict()
    axis_dict=OrderedDict()
    for mr in [mr1k4_soms, mr2k4_kbo,mr3k4_kbo, mr4k4_kbo, mr5k4_kbo]:
        for keys in dir(mr):
            axis = getattr(mr, keys)
            if hasattr(axis, 'user_readback'):
                if axis.connected:
                    curr_pos[axis.name]=axis.position
                    axis_dict[axis.name]=axis
                    pv=axis.user_readback.pvname
                    arch_data = archive.get(pv, xarray=True, start=tstamp, end=tstamp+datetime.timedelta(minutes=1))
                    prev_pos[axis.name] = arch_data.sel(time=tstamp, method='nearest').to_dataframe()[pv]['vals']
       
    settings = {'Axis':curr_pos.keys(), 
                'Current':curr_pos.values(), 
                'Archive':prev_pos.values(), 
                'Difference':np.array(list(curr_pos.values()))-np.array(list(prev_pos.values()))}            
    settingsDF_str = tabulate(pd.DataFrame(settings), headers="keys", tablefmt="pretty")
    #print(pd.DataFrame(settings))
    print(settingsDF_str)
    

def get_slits(slits=sl1k4, log=False, msg=None, update_preset=False, include_cam=None, **kwargs):
    """
    Get the current position of slits in the beamline, and optionally add to the elog. 

    Parameters
    ----------
    slits : slit object or list of slit objects, optional
        Slits to log: : sl1k4 (default), sl2k0, sl1k4, sl2k4
    log : bool, optional
        boolean to log the positions in the elog. The default is False.
    msg : str, optional
        optional argument to append a message to the elog with the fzp positions. The default is None.
    update_preset : bool, optional
        update the tmo preset positions for sl2k0 and sl1k4 to the current position. 
    include_cam : list of str, optional
        list of camera names to append to elog post.
    tag : str, optional
        List of tags to add to an elog post, by default the slit name is tagged
    """
    
    def _update_presets(slits, preset='tmo'):
        for slit in slits:
            if slit.name == 'sl1k0':
                logger.error('no preset for sl1k0')
                return None
            else:
                old_pos=OrderedDict()
                new_pos=OrderedDict() 
                for slits in [sl2k0,sl1k4]:
                    for axis in ['north','south','top','bottom']:
                        motor = getattr(slits, axis)
                        motor.presets.sync()
                        if hasattr(motor.presets.positions, preset):
                            preset_pos = getattr(motor.presets.positions, preset)
                            old_pos[motor.name] = np.copy(preset_pos.info.get('value'))
                            preset_pos.update_pos()
                            new_pos[motor.name] = preset_pos.info.get('value')
                        else:
                            old_pos[motor.name] = ''
                            motor.presets.add_here_hutch(preset)
                            preset_pos = getattr(motor.presets.positions, preset)
                            new_pos[motor.name] = preset_pos.info.get('value')
                        updateDF = pd.DataFrame({"Axis:": old_pos.keys(), "Old Preset:": old_pos.values(), "New Preset:": new_pos.values()})
                        update_msg = ("Update %s preset positions for slits. \n" % preset) + updateDF.to_string()
                return update_msg
    
    curr_positions= {}
    if not isinstance(slits, list):
        slits = [slits]
    for slit in slits:
        for keys in dir(slit):
            motor = getattr(slit, keys)
            if hasattr(motor, 'position'):
                curr_positions[motor.name] = getattr(motor, 'position')
        kwargs['tags'] = kwargs.get('tags', ' ') + ' ' + slit.name        
    if update_preset:
        if isinstance(update_preset, bool): 
            update_preset = 'tmo'
        update_msg = _update_presets(slits, preset=update_preset)
        if update_msg is not None:
            log=True
            kwargs['tags'] = kwargs.get('tags', ' ') + ' ' + 'update_preset'
            kwargs['facility'] = False #True  #Write to instrument elog
            if (msg is not None):
                msg = msg + "\n" + update_msg
            else:
                msg = update_msg
    _log_positions(curr_positions, log=log, msg=msg, include_cam=include_cam, **kwargs)

def reset_sl1k0(move_centers=True):
    """
    Reset to the maximum size allowed by PMPS based on the apertures. 

    Parameters
    ----------
    move_centers : bool, optional
        Moves the centers to facilitate the largest gap. The default is True.

    """
    pmps_dict={'5.5mm': 2.95, '8mm': 6.95, '10mm': 8.45, '13mm': 8.75}
    key = at1k0.apperture_array.maa_01.position
    get_slits(sl1k0, log=True, msg='SL1K0 slit position before adjusting')
    
    vo = sl1k0.vo.wm()
    ho = sl1k0.ho.wm()
    
    if move_centers:
        if vo > 0.015: sl1k0.vo.umv(0)
        if ho > 0.015: sl1k0.ho.umv(0)
        sl1k0.vg.umv(pmps_dict[key])
        sl1k0.hg.umv(pmps_dict[key])
    else:    
        sl1k0.vg.umv(pmps_dict[key]-2*vo)
        sl1k0.hg.umv(pmps_dict[key]-2*ho)
        
    get_slits(sl1k0, log=True, msg='new SL1K0 slit positions after moving')

def reset_slits(preset='tmo', home=False):
    """ Reset slits to previous saved position. home function will home SL1K4."""
    
    get_slits([sl2k0,sl1k4], log=True, msg='current slit positions before moving to %s preset.' % preset)
    axes = ['north','south','top','bottom']
    
    def _wait_for_move():
        time.sleep(1)
        done_moving = False
        while not done_moving:
            done_moving = True
            for slits in [sl2k0,sl1k4]: 
                for axis in axes:
                    motor = getattr(slits, axis)
                    done_moving = done_moving and (not motor.moving) 
         
    if home: #home sl1k4
        for axis in axes:
            motor = getattr(sl1k4, axis)
            motor.home(wait=False)
            logger.debug('homing %s' % motor.name)
#        done_moving = False
        logger.info('homing sl1k4')
#        time.sleep(1)
#        while not done_moving:
#            done_moving = True
#            for axis in axes:
#                motor = getattr(sl1k4, axis)
#                done_moving = done_moving and (not motor.moving) 
        _wait_for_move()
        #clear any errors
        time.sleep(1)
        logger.debug('clearing errors')
        for axis in axes:
            motor = getattr(sl1k4, axis)
            motor.clear_error()
        logger.info('homing complete')
        
    #move slits to presets    
    for slits in [sl2k0,sl1k4]:    
        for axis in axes:
            motor = getattr(slits, axis)
            motor.presets.sync()
            motor.clear_error() #clear any errors before moving
            mv = getattr(motor, 'mv_'+preset)
            mv()        
    
    #check all slits are done moving
    _wait_for_move()
#    time.sleep(1)
#    done_moving = False
#    while not done_moving:
#        done_moving = True
#        for slits in [sl2k0,sl1k4]: 
#            for axis in axes:
#                motor = getattr(slits, axis)
#                done_moving = done_moving and (not motor.moving) 
    
    #hacky fix to a problem with using the presets on each motor and removing old values from memory.
    #this writes the current position to the set position.
    for slits in [sl2k0,sl1k4]:
        pos = {}
        for axis in ['vg','vo','hg','ho']:
            motor = getattr(slits,axis)
            pos.update({axis:motor.wm()})
        for i in range(2):
            for axis in ['vg','hg','vo','ho']:
                motor = getattr(slits,axis)
                #motor.mv(pos[key])
                motor.setpoint.put(pos[axis])
            time.sleep(1)   

    #check all slits are done moving
    _wait_for_move()
#    done_moving = False
#    while not done_moving:
#        done_moving = True
#        for slits in [sl2k0,sl1k4]: 
#            for axis in axes:
#                motor = getattr(slits, axis)
#                done_moving = done_moving and (not motor.moving) 
            
    #log changes to the slits
    get_slits([sl2k0,sl1k4], log=True, msg=('Moved slits to previous %s preset position' % preset))
#    reset_sl1k0()

def open_slits(gaps=None):
    """
    Log the current positions and open sl1k4 and sl2k0 slits. 

    Parameters
    ----------
    gaps : list of floats, optional
        List of slit gaps [sl2k0, sl1k4]. The default is [10,14].
    
    """

    if gaps is not None:
        get_slits([sl2k0,sl1k4], log=True, msg='current slit positions before opening.')
        sl2k0.hg.umv(gaps[0])
        sl2k0.vg.umv(gaps[0])
        sl1k4.hg.umv(gaps[1])
        sl1k4.vg.umv(gaps[1])
    else:
        reset_slits(preset='open')
    
def tweak_slits(gap=None, x=None, y=None):
    """
    Adjust SL2K0 and SL1K4 together. 

    Parameters
    ----------
    gap : float, optional
        Value to increase/decrease the gap size. The default is None.
    x : float, optional
        Value to shift the horizontal offset. The default is None.
    y : float, optional
        Value to shift the vertical offset. The default is None.

    """
    
    if gap is not None:
        sl2k0.hg.mvr(gap)
        sl2k0.vg.mvr(gap)
        sl1k4.hg.mvr(gap)
        sl1k4.vg.mvr(gap)

    if x is not None:
        sl2k0.ho.mvr(x)
        sl1k4.ho.mvr(x)
        
    if y is not None:
        sl2k0.vo.mvr(y)
        sl1k4.vo.mvr(y)
               
def get_fzp(log=False, msg=None, include_cam=None, **kwargs):
    """
    Get current position of the fzp motors and intensifier settings. 
     
    """
    curr_positions= OrderedDict()
    set_points = OrderedDict()
    for keys in dir(sp1k4):
        motor = getattr(sp1k4, keys)
        if hasattr(motor, 'position'):
            curr_positions[keys] = getattr(motor, 'position')
            if hasattr(motor, 'user_setpoint'):
                set_points[keys] = getattr(motor, 'user_setpoint').get()
            else:
                set_points[keys] = getattr(motor, 'position')
    curr_positions[pf1k4.z_motor.name] = pf1k4.z_motor.position
    set_points[pf1k4.z_motor.name] = pf1k4.z_motor.user_setpoint.get()
    if sp1k4_intensifier.connected:
        curr_positions['Intensifier MCP [V]'] = sp1k4_intensifier.mcp_voltage.get()
        curr_positions['Intensifier Gate [us]'] = sp1k4_intensifier.gate_time.get()/1000
        set_points['Intensifier MCP [V]'] = sp1k4_intensifier.mcp_voltage.setpoint
        set_points['Intensifier Gate [us]'] = sp1k4_intensifier.gate_time.setpoint/1000
    else:
        curr_positions['Intensifier MCP [V]'] = 'n/c'
        curr_positions['Intensifier Gate [us]'] = 'n/c'
        set_points['Intensifier MCP [V]'] = 'n/c'
        set_points['Intensifier Gate [us]'] = 'n/c'
    
    if sp1k4_tpr.connected:
        curr_positions['Intensifier TPR Delay [us]'] = sp1k4_tpr.delay_ticks.get()*(1/185.714286E-3)/1000
        curr_positions['Intensifier Trig Rate'] = sp1k4_tpr.eventrate.get()
        set_points['Intensifier TPR Delay [us]'] =  sp1k4_tpr.delay_setpoint.get()/1000
        set_points['Intensifier Trig Rate'] = ''
    else:
        curr_positions['Intensifier TPR Delay [us]'] = 'n/c'
        curr_positions['Intensifier Trig Rate'] = 'n/c'
        set_points['Intensifier TPR Delay [us]'] =  'n/c'
        set_points['Intensifier Trig Rate'] = 'n/c'
    
    kwargs['tags'] = kwargs.get('tags', ' ') + ' fzp'

    if include_cam == 'default' or True:
        include_cam=['sp1k4-gige', 'im5k4', 'sp1k4_yag_f']
    
    _log_positions(curr_positions, log=log, msg=msg, include_cam=include_cam, set_points=set_points, **kwargs)
get_fzp.__doc__ += _log_positions.__doc__

class SP1K4yagLens(SyncAxisTmo): 
    lens_x = OCpt(sp1k4.thorlab_lens_x)
    yag_x = OCpt(sp1k4.yag_x)
    tab_component_names = True
    tab_whitelist=['set_offset_to_current_position']
#    pos = sync * scale + offset
    scales = {'yag_x':1, 'lens_x': 0.9797} #maybe scale should be 1.0272?
    offsets = {'yag_x': 0,'lens_x': -5.08979}

    #Data from MFL to calibrate the scale and offset
    #yag = np.array([2.12,3.12,4.138,1.108,0.058,-0.968,-1.993,-3.018])
    #lens = np.array([-3,-2,-1.05,-4.04,-5.04,-6.039,-7.039,-8.038])
    
    warn_deadband = 0.005
    fix_sync_keep_still = 'yag_x'
    sync_limits = (-100, 100)
   
    
#    def set_offset_to_current_position(self):
#        """ Take the current positions and adjust the offsets to re-sync"""
#        fix_sync_pos = getattr(self, self.fix_sync_keep_still).position
##        fix_sync_pos = getattr(self, self.fix_sync_keep_still).position * self.scales[self.fix_sync_keep_still] + self.offsets[self.fix_sync_keep_still]
#        curr_pos = {}
#        for attr in self._real_attrs:
#            curr_pos[attr] = getattr(self, attr).position
#            self.offsets[attr] = curr_pos[attr] - fix_sync_pos * self.scales[attr]
#        
##        curr_lens = self.lens_x.wm()
##        curr_yag = self.yag_x.wm()
##        new_offset = curr_lens - curr_yag * self.scales['lens_x']
##        self.offsets['lens_x'] = new_offset
#        self._setup_offsets()
        
sp1k4_yag_focus = SP1K4yagLens('', name='sp1k4_yag_focus')

def fix_wfs():
    """
    Adjusts the binning of image1 ROI for im5k4 camera. The WFS expects no binning. 

    Returns
    -------
    None.

    """
#    bin_x = camviewer.im5k4.image1_roi.bin_.x.get()
#    bin_y = camviewer.im5k4.image1_roi.bin_.y.get()
    bin_x = im5k4.detector.bin_x.get()
    bin_y = im5k4.detector.bin_y.get()
    
    logger.info('current binning for WFS is: [%(x)d, %(y)d], setting to [1,1]', {'x':bin_x, 'y':bin_y})
    
    im5k4.detector.bin_x.put(1)
    im5k4.detector.bin_y.put(1)

class TriCamIntensifier(BaseInterface, Device):
    """
    Class for controlling the TRiCAM Image Intensifier for the TMO FZP spectrometer.
    """
    tab_component_names = True
    tab_whitelist = ['enable','disable']

    intensifier_power = Cpt(EpicsSignal, ':GetIntensifierPowerStatus', write_pv=':PowerIntensifierOnOFF', kind='hinted', string=True,
                            doc='Intersifier On/Off State: [0] Disable, [1] Enable')
    mcp_voltage = Cpt(EpicsSignal, ':GetMCPVoltage', write_pv=':SetMCPVoltage', kind='hinted',
                      doc='MCP voltage')
    current_mode = Cpt(EpicsSignal, ':GetCurrentMonitorMode', write_pv=':SetCurrentMonitorMode', kind='omitted', string=True,
                       doc='Current monitor mode')
    current_shutdown = Cpt(EpicsSignal, ':GetCurrentShutdownLevel', write_pv=':SetCurrentShutdownLevel', kind='omitted',
                           doc='shutdown current level.')
    current_level = Cpt(EpicsSignal, ':GetCurrentShutdownLevel', write_pv=':SetCurrentShutdownLevel', kind='omitted',
                        doc='current shutdown level')
    default_level = Cpt(EpicsSignalRO, ':GetCurrentDefaultLevel', kind='omitted',
                        doc='default current shutdown level')
    current = Cpt(EpicsSignalRO, ':GetCurrentLevel', kind='normal', 
                  doc='current level in uA.')
    trigger_mode = Cpt(EpicsSignal, ':GetTriggerGateMode', write_pv=':SetTriggerGateMode', kind='hinted', string=True,
                     doc='[0] Disable triggering, [1] Sync with camera exposure, [2] Follow camera exposure, [3] Rising edge sync, [4] Follow Sync input, [5] Software trigger, [6] Falling edge sync, [7] Fixed frequency')
    gate_enable = Cpt(EpicsSignal, ':IsGateEnabled', write_pv=':EnableGate', kind='hinted', string=True,
                    doc='gate enable: [0] Disable, [1] Enable')
    gate_time = Cpt(EpicsSignal, ':GetGateOpenTime', write_pv=':SetGateOpenTime', kind='hinted',
                    doc='gate open time [ns]')
    delay = Cpt(EpicsSignal, ':GetGateDelayTime', write_pv=':SetGateDelayTime', kind='omitted',
                doc='gate delay time [ns]')    
    clock = Cpt(EpicsSignal, ':GetClockScaler', write_pv=':SetClockScaler', kind='omitted',
                doc='clock scaaler')    
    sn = Cpt(EpicsSignalRO, ':GetIntensifierSN', kind='omitted')
    fwv = Cpt(EpicsSignalRO,':GetFirmwareVersion', kind='omitted')
    
    def enable(self, voltage=450, mode=3, gate_time=1E4):
        self.mcp_voltage.put(450)
        self.intensifier_power.put(1)
        self.trigger_mode.put(mode)
        self.gate_enable.put(1)
        self.gate_time.put(gate_time)
        self.mcp_voltage.put(voltage)
        
    def disable(self):
        self.mcp_voltage.put(450, wait=True)
        self.intensifier_power.put(0)

sp1k4_intensifier = TriCamIntensifier('TMO:TRICAM:01', name='sp1k4_intensifer')

sp1k4_tpr = TprTrigger('TMO:MISC:TPR:01', channel=0, timing_mode=TimingMode.LCLS2, name='sp1k4_tpr')

#def update_fzp_target(set_pos=True):
#    """"resets the current position to the setpoint for the current target."""
#    state_num = sp1k4.zone_plate.state.get()
#    
#    state_update={'state_name': sp1k4.zone_plate.position, 'state_num':state_num}
#    
#    for i, axis in enumerate(['x','y','z']):
#        m_config = getattr(sp1k4.zone_plate.config, ('m%s_state%02g' % (i+1, state_num)) )
#        motor = getattr(sp1k4, ('zone_plate_%s' % axis) )
#        state_update[('previous m%s'%(i+1))] = m_config.setpoint.get()
#        state_update[('new m%s'%(i+1))] = motor.position
#        if set_pos:
#            m_config.setpoint.put(motor.position)
#    _log_positions(state_update, log=set_pos, msg='update sp1k4 target positions', tags=['sp1k4', 'states', 'update'])
#    return
        
class FzpDet(BaseInterface, Device):
    def __init__(self, **kwargs):
        self.uid = None
        self.cam = getattr(camviewer, 'sp1k4-gige')
        self.roi = (self.cam.image1_roi.min_xyz.min_x.get(),
                    self.cam.image1_roi.size.x.get(),
                    self.cam.image1_roi.bin_.x.get(),
                    self.cam.image1_roi.min_xyz.min_y.get(),
                    self.cam.image1_roi.size.y.get(),
                    self.cam.image1_roi.bin_.y.get(),
                   )
        return super().__init__(**kwargs)
    
    def describe(self):
        desc = self.cam.image1.array_data.describe()
        desc['fzp_data'] = desc.pop(self.cam.image1.array_data.name)
        desc['fzp_axis'] = {'source':'fixed', 'dtype':'array','shape':[2048], 'units':'adu'}
        return desc
    
    def _data_cb(self, value, timestamp, **kwargs):
        prev = self.data['fzp_data'].get('value')
        self.data['fzp_data'].update({'value': prev + value,
                                          'timestamp':timestamp})
        self._norm+=1
        
    def stage(self):
        self.data=OrderedDict()
        self.data['fzp_data']={'value':np.zeros(2048), 'timestamp':0}
        self.data['fzp_axis']={'value':range(2048), 'timestamp':0}        
        self.cam.image1_roi.min_xyz.min_x.set(self.roi[0])
        self.cam.image1_roi.size.x.set(self.roi[1])
        self.cam.image1_roi.bin_.x.set(self.roi[2])
        self.cam.image1_roi.min_xyz.min_y.set(self.roi[3])
        self.cam.image1_roi.size.y.set(self.roi[4])
        self.cam.image1_roi.bin_.y.set(self.roi[5])       
        self.uid = self.cam.image1.array_data.subscribe(self._data_cb)    
        return [self]
    
    def trigger(self):
        self.data.update({'fzp_data':{'value':np.zeros(2048), 'timestamp':0}})
        self._norm = 0
        return DeviceStatus(self, done=True, success=True)
            
    def read(self):
        self.data['fzp_data'].update({'value':self.data['fzp_data'].get('value')/self._norm})
        return self.data
    
    def unstage(self):
        self.cam.image1.array_data.unsubscribe(self.uid)
        return [self]

#ori_roi=(0,2048,1,1164,160,160)
fzp_det = FzpDet(name='sp1k4_gige')
    
     

