import numpy as np
from pcdsdevices.device import ObjectComponent as OCpt
from pcdsdevices.pseudopos import SyncAxis
from tmo.db import mr2k4_kbo, mr3k4_kbo, mr1k4_soms, sp1k4, mr4k4_kbo, mr5k4_kbo, sl1k0, sl2k0, sl1k4, sl2k4, rtdsk0, camviewer
from tmo.db import elog
from tmo.tmo_beamline_control import _log_positions
from collections import OrderedDict
import pandas as pd
import logging
logger = logging.getLogger(__name__)

#from tmo.mirror_check import MirrorCheck
#from tmo.mirrors import mirrors

def get_KB1(log=False, msg=None, save_bender_preset=None, preset_type='exp', **kwargs):
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
        Str to specify if the preset is associated with the 'hutch' or the current 'exp'eriment. The default is 'exp'.
    """
    curr_positions= {}
    for keys in dir(mr2k4_kbo):
        motor = getattr(mr2k4_kbo, keys)
        if hasattr(motor, "position"):
            curr_positions["mr2k4 "+keys] = getattr(motor, 'position')
    for keys in dir(mr3k4_kbo):
        motor = getattr(mr3k4_kbo, keys)
        if hasattr(motor, "position"):
            curr_positions["mr3k4 "+keys] = getattr(motor, 'position')  
    kwargs['tags'] = 'kbo1 ' + kwargs.get('tags', ' ')  

    if save_bender_preset is not None:
        log = True
        if preset_type == 'exp':
            mr2k4_kbo.bender_us.presets.add_here_exp(name=save_bender_preset, comment=msg)
            mr2k4_kbo.bender_ds.presets.add_here_exp(name=save_bender_preset, comment=msg)
            mr3k4_kbo.bender_us.presets.add_here_exp(name=save_bender_preset, comment=msg)
            mr3k4_kbo.bender_ds.presets.add_here_exp(name=save_bender_preset, comment=msg)
            kwargs['tags'] = kwargs.get('tags', ' ') + ' presets'
            msg = 'Update ' + save_bender_preset + ' for experiment with current bender positions. \n' + (msg or '')
        elif preset_type == 'hutch':
            mr2k4_kbo.bender_us.presets.add_here_hutch(name=save_bender_preset, comment=msg)
            mr2k4_kbo.bender_ds.presets.add_here_hutch(name=save_bender_preset, comment=msg)
            mr3k4_kbo.bender_us.presets.add_here_hutch(name=save_bender_preset, comment=msg)
            mr3k4_kbo.bender_ds.presets.add_here_hutch(name=save_bender_preset, comment=msg)
            kwargs['tags'] = kwargs.get('tags', ' ') + ' presets'
            msg = 'Update ' + save_bender_preset + ' for hutch with current bender positions. \n' + (msg or '')
        else:
            logger.error('preset_type not recognizied. Should be "exp" or "hutch".')

    _log_positions(curr_positions, log, msg, **kwargs) 

def set_KB1(bender_preset=None):
    """
    Move the KB1 benders to a preset position. Passing no position will return the list of existing presets.

    Parameters
    ----------
    bender_preset : str, optional
        name of preset position. The default is None.
    """
    if bender_preset is not None:
        if ( hasattr(mr2k4_kbo.bender_us.presets.positions, bender_preset) and 
             hasattr(mr2k4_kbo.bender_ds.presets.positions, bender_preset) and
             hasattr(mr3k4_kbo.bender_us.presets.positions, bender_preset) and
             hasattr(mr3k4_kbo.bender_ds.presets.positions, bender_preset) ):
            mv1=getattr(mr2k4_kbo.bender_us, ('umv_'+bender_preset))
            mv2=getattr(mr2k4_kbo.bender_ds, ('umv_'+bender_preset))
            mv3=getattr(mr3k4_kbo.bender_us, ('umv_'+bender_preset))
            mv4=getattr(mr3k4_kbo.bender_ds, ('umv_'+bender_preset))
            mv1()
            mv2()
            mv3()
            mv4()
            get_KB1(log=True, msg=('Move KB1 to preset position '+bender_preset))
        else:
            logger.error('One of the motors is missing the desired state.')
    else:
         
        state_dict={"State:": [], mr2k4_kbo.bender_us.name: [], mr2k4_kbo.bender_ds.name: [], mr3k4_kbo.bender_us.name: [], mr3k4_kbo.bender_ds.name: []}
        for state in (list(vars(mr2k4_kbo.bender_us.presets.positions).keys()) 
                      + list(vars(mr2k4_kbo.bender_ds.presets.positions).keys()) 
                      + list(vars(mr3k4_kbo.bender_us.presets.positions).keys()) 
                      + list(vars(mr3k4_kbo.bender_ds.presets.positions).keys()) ):
            if state not in state_dict["State:"]:
                state_dict["State:"].append(state)
                state_dict[mr2k4_kbo.bender_us.name].append(vars(mr2k4_kbo.bender_us.presets.positions).get(state, ' '))
                state_dict[mr2k4_kbo.bender_ds.name].append(vars(mr2k4_kbo.bender_ds.presets.positions).get(state, ' '))
                state_dict[mr3k4_kbo.bender_us.name].append(vars(mr3k4_kbo.bender_us.presets.positions).get(state, ' '))
                state_dict[mr3k4_kbo.bender_ds.name].append(vars(mr3k4_kbo.bender_ds.presets.positions).get(state, ' '))
        print (pd.DataFrame(state_dict))
            
        
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
        if hasattr(motor, "position"):
            curr_positions["mr1k4 "+keys] = getattr(motor, 'position')
    kwargs['tags'] = kwargs.get('tags', ' ') + ' mr1k4'        
    _log_positions(curr_positions, log, msg, **kwargs)
        
def get_slits(slits=sl1k4, log=False, msg=None, **kwargs):
    """
    Get the current position of slits in the beamline, and optionally add to the elog. 

    Parameters
    ----------
    slits : slit object, optional
        Slits to log: : sl1k4 (default), sl2k0, sl1k4, sl2k4
    log : bool, optional
        boolean to log the positions in the elog. The default is False.
    msg : str, optional
        optional argument to append a message to the elog with the fzp positions. The default is None.
    tag : str, optional
        List of tags to add to an elog post, by default the slit name is tagged
    """
    curr_positions= {}
    for keys in dir(slits):
        motor = getattr(slits, keys)
        if hasattr(motor, 'position'):
            curr_positions[motor.name] = getattr(motor, 'position')
#    curr_positions= {
#     slits.name+" x-center":slits.xcenter.position,
#     slits.name+" x-width":slits.xwidth.position,
#     slits.name+" y-center":slits.ycenter.position,
#     slits.name+" y-width":slits.ywidth.position,
#    }
    kwargs['tags'] = kwargs.get('tags', ' ') + ' ' + slits.name        
    _log_positions(curr_positions, log, msg, **kwargs)    

def get_fzp(log=False, msg=None, **kwargs):
    """
    Get current position of the fzp motors.
    log - boolean to log the positions in the elog 
    msg - optional argument to append a message to the elog with the fzp positions 
    """
    curr_positions= {}
    for keys in dir(sp1k4):
        motor = getattr(sp1k4, keys)
        if hasattr(motor, 'position'):
            curr_positions[keys] = getattr(motor, 'position')
    kwargs['tags'] = kwargs.get('tags', ' ') + ' fzp'        
    _log_positions(curr_positions, log, msg, **kwargs)

def fix_wfs():
    """
    Adjusts the binning of image1 ROI for im5k4 camera. The WFS expects no binning. 

    Returns
    -------
    None.

    """
    bin_x = camviewer.im5k4.image1_roi.bin_.x.get()
    bin_y = camviewer.im5k4.image1_roi.bin_.y.get()
    
    logger.info('current binning for WFS is: [%(x)d, %(y)d], setting to [1,1]', {'x':bin_x, 'y':bin_y})
    
    camviewer.im5k4.image1_roi.bin_.x.put(1)
    camviewer.im5k4.image1_roi.bin_.y.put(1)

#class SP1K4ymc(SyncAxis):
#    lens_x = OCpt(sp1k4.lens_x)
#    yag_x = OCpt(sp1k4.yag_x)
#    tab_component_names = True
#    scales = {'lens_x': 1}
#    warn_deadband = 0.01
#    fix_sync_keep_still = 'yag_x'
#    sync_limits = (-100, 100)
#
#sp1k4_ymc = SP1K4ymc('', name='sp1k4_ymc')

#Moved to mirror_check.py

'''
def check_kbo1_transmission(output_path ="/reg/g/pcds/pyps/apps/hutch-python/tmo/tmo/mirror-check", scan_duration = 60.0):
     """
     perform transmission check for TMO KBO1 mirror using im3k4 and im4k4 power meters.

     Parameters
     ----------
     output_path : str, optional
         DESCRIPTION. The default is "/reg/g/pcds/pyps/apps/hutch-python/tmo/tmo/mirror-check".
     scan_duration : float, optional
         DESCRIPTION. The default is 60.0.

     Returns
     -------
     None.

     """
     kbo1=MirrorCheck(mirrors["TMO_KBO1"], output_path, scan_duration)
     kbo1.start_test()

def check_kbo2_transmission(output_path ="/reg/g/pcds/pyps/apps/hutch-python/tmo/tmo/mirror-check", scan_duration = 60.0):
     """
     perform transmission check for TMO KBO2 mirror using im5k4 and im6k4 power meters.

     Parameters
     ----------
     output_path : str, optional
         DESCRIPTION. The default is "/reg/g/pcds/pyps/apps/hutch-python/tmo/tmo/mirror-check".
     scan_duration : float, optional
         DESCRIPTION. The default is 60.0.

     Returns
     -------
     None.

     """
     kbo2=MirrorCheck(mirrors["TMO_KBO2"], output_path, scan_duration)
     kbo2.start_test()
'''