from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from pcdsdevices.pv_positioner import PVPositionerIsClose
from pcdsdevices.epics_motor import SmarActTipTilt, EpicsMotorInterface, SmarAct
from ophyd.device import Component as Cpt
from ophyd.signal import EpicsSignal, EpicsSignalRO
from tmo.db import li1k4, tm1k4
from tmo.db import elog
from tmo.tmo_sequencer import tmo_evr2
from tmo.tmo_beamline_control import _log_positions
import pandas as pd
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

#The following method has been replaced with a new version. 
class TmoLaserOld: 
    """
    This class controls motors associated with the IP1 laser system in TMO.
    stop method closes the laser shutter
    enable method will open the laser shutter
    """
    def __init__(self):
        self.lens_x = EpicsMotorInterface('LM1K4:EJX_MP2_LS3_LM2', name='lens_x')
        self.lens_y = EpicsMotorInterface('LM1K4:EJX_MP2_LS3_LM3', name='lens_y')
        self.lens_z = EpicsMotorInterface('LM1K4:EJX_MP2_LS3_LM1', name='lens_z')
        self.inj_mr = EpicsMotorInterface('LM1K4:INJ_DP2_MR1', name='inj_mr')
        self.comp_sep = EpicsMotorInterface('LM1K4:COM_MP1_GT2_LM1', name='grating_sep')
        self.atm_mr_tt = SmarActTipTilt(prefix="LM1K4:ATM_MP1_MR1", tip_pv="_TIP1", tilt_pv="_TILT1", name="atm_mr_tt")
        self.oap_tt1 = SmarActTipTilt(prefix="LM1K4:EJX_MP1_OAP1_MR1", tip_pv="_TIP", tilt_pv="_TILT", name="oap_tt1")
        self.oap_z = EpicsMotorInterface('LM1K4:EJX_MP1_OAP1_LM1', name='oap_z')
        self.oap_tt3 = SmarActTipTilt(prefix="LM1K4:EJX_MP1_OAP1_MR3", tip_pv="_TIP", tilt_pv="_TILT", name="oap_tt3")
        self.wp = WaveplateMotor('LM1K4:EJX_MP2_ATT1_POL1:ELL:M1', name='WP')
    def stop(self):
        tmo_evr2.polarity.set(0)
        tmo_evr2.eventcode.set(0)
        tmo_evr2.enable()
        #caput("TMO:USR:BHC:AOUT:1", 0)
    def enable(self):
        tmo_evr2.polarity.set(1)
        tmo_evr2.eventcode.set(69)
        tmo_evr2.enable()
        #caput("TMO:USR:BHC:AOUT:1", 5)


class LI3K4(BaseInterface, Device):
    """
    This class controls the dual carriage smaract stages and uses motion presets to make correlated movements of of the carriages..
    """
    tab_component_names= True
    tab_whitelist = ['position','set_current_position','list_states']
    
    mr1 = Cpt(SmarAct, '_MR1', kind='normal', doc='first mirror on dual carriage smarAct')
    mr2 = Cpt(SmarAct, '_MR2', kind='normal', doc='second mirror on dual carriage smarAct')
    
    def position(self,state=None):
        """
        Calling the object will either move the object or get the current position, depending on the state arguement.
        preset positions can be defined with the set_current_position method, or using the individual preset class for the two motors. 
        """
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
        print(self.mr1.presets.positions)
        print(self.mr2.presets.positions)

class TmoLaser(BaseInterface, Device):
    """
    This class controls motors associated with the IP1 laser system in TMO.
    stop method closes the laser shutter
    enable method will open the laser shutter
    """
    tab_component_names= True
    tab_whitelist = ['stop', 'enable','laser_status']
        
    lens_x = Cpt(EpicsMotorInterface, 'LM1K4:EJX_MP2_LS1_LM2', kind='normal')
    lens_y = Cpt(EpicsMotorInterface, 'LM1K4:EJX_MP2_LS1_LM3', kind='normal')
    lens_z = Cpt(EpicsMotorInterface, 'LM1K4:EJX_MP2_LS1_LM1', kind='normal')
    inj_mr = Cpt(SmarAct, 'LM1K4:INJ_DP2_MR2', kind='normal')
#    comp_sep = Cpt(SmarAct, 'LM1K4:COM_MP1_GT2_LM1', kind='normal')
    atm_mr_tt = Cpt(SmarActTipTilt, prefix='LM1K4:ATM_MP1_MR3', tip_pv="_TIP", tilt_pv="_TILT", kind='normal')
    oap_tt1 = Cpt(SmarActTipTilt, prefix="LM1K4:EJX_MP1_OAP1_MR1", tip_pv="_TIP", tilt_pv="_TILT", kind='normal')
    oap_z = Cpt(SmarAct, 'LM1K4:EJX_MP1_OAP1_LM1', kind='normal')
    oap_tt3 = Cpt(SmarActTipTilt, prefix="LM1K4:EJX_MP1_OAP1_MR3", tip_pv="_TIP", tilt_pv="_TILT", kind='normal')
    wp = Cpt(WaveplateMotor, 'LM1K4:EJX_MP2_ATT1_POL1:ELL:M1', kind='normal')
#    delay = Cpt(LaserDelayA, 'LAS:LHN:LLG2:01:PHASCTL', egu='ns', kind='omitted')
#    delay = Cpt(LaserDelayB, 'LAS:LHN:LLG2:01:PHASCTL:RF_PHASE', egu='ns', kind='omitted')
#    li3k4_mr1 = Cpt(SmarAct, 'LM1K4:EJX_DP6_CVS1_MR1', kind='normal')
#    li3k4_mr2 = Cpt(SmarAct, 'LM1K4:EJX_DP6_CVS1_MR2', kind='normal')
    li3k4 = Cpt(LI3K4, 'LM1K4:EJX_DP6_CVS1', kind='normal')
    
    def stop(self):
        logger.error('Cannot stop the laser.')
        
    def enable(self):
        logger.error('Cannot start the laser.')
        
    def laser_status(self, log=False, msg=None, run=None):
        curr_positions= {}
        for key in dir(self):
            motor = getattr(self, key)
            if hasattr(motor, 'connected'):
                if motor.connected:
                    if hasattr(motor, "position"):
                        position = getattr(motor, 'position')
                        if isinstance(position, float):
                            curr_positions[key] = position
                        else:
                            for keys in dir(motor):
                                mot = getattr(motor, keys)
                                if hasattr(mot, 'position'):
                                    position = getattr(mot, 'position')
                                    if isinstance(position, float):
                                        curr_positions[keys] = position
                                    
        _log_positions(curr_positions, log, msg=msg, tags='laser', run=run) 

tmo_laser = TmoLaser(name='tmo_laser')
stop_laser = tmo_laser.stop


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
       "QWP":tmo_laser.wp.position,
    }
    kwargs['tags'] = kwargs.get('tags', ' ') + ' li2k4'
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
    kwargs['tags'] = kwargs.get('tags', ' ') + ' li1k4'
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
     "ATM MR tip":tmo_laser.atm_mr_tt.tip.total_step_count.get(),
    "ATM MR tilt":tmo_laser.atm_mr_tt.tilt.total_step_count.get(),
#    "ATM target":tm1k4.target.position,
#    "ATM y-Motor":tm1k4.y_motor.position
    }
    for keys in dir(tm1k4):
        motor = getattr(tm1k4, keys)
        if hasattr(motor, "position"):
            curr_positions[motor.name] = motor.position           
    kwargs['tags'] = kwargs.get('tags', ' ') + ' atm'
    _log_positions(curr_positions, log, **kwargs)
  