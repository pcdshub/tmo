from ophyd.device import Component as Cpt
from ophyd.device import FormattedComponent as FCpt
from ophyd.signal import EpicsSignal, EpicsSignalRO
from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from pcdsdevices.pv_positioner import PVPositioner, PVPositionerDone
from tmo.tmo_utils import SyncAxisTmo

import numpy as np

def delay_line_steps(pitch_step = 3e-5, roll_step = 3e-5):
    'steps are in rad, returns pitch motor 1, pitch motor 2, y offset'
    pitch_motor_1_init = -20.79 # in mm, pitch motor 1, NAME: pos x POSITION: neg x
    pitch_motor_2_init = -18.53 # in mm, pitch motor 2, NAME: neg x POSITION: pos x
    y_init = -72.9 # in mm
    
    pitch_motor_1_lims = (-21.3, -20.52) # lower lim, upper lim
    pitch_motor_2_lims = (-19.19, -18.2) # lower lim, upper lim
    y_lims = (-73.9, -71.9) # lower lim, upper lim
    
    pitch_motor_1_steps = []
    pitch_motor_2_steps = []
    y_steps = []
    
    pitch_motor_ratio = 0.37
    
    # loop over pitch
    for pitch_offset in np.arange(-5e-4, 5e-4, pitch_step): # in rad
        
        pitch_motor_1_i = pitch_motor_1_init - np.sin(pitch_offset) * 680
        pitch_motor_2_i = pitch_motor_2_init - np.sin(pitch_offset) * 680
    
        for roll_offset in np.arange(-5e-4, 5e-4, roll_step):
            
            delta_pitch = np.arcsin(roll_offset) * 580
            
            delta_pitch_motor_1_j = delta_pitch * pitch_motor_ratio
            delta_pitch_motor_2_j = delta_pitch * (1 - pitch_motor_ratio)
            
            pitch_motor_1_j = pitch_motor_1_i - delta_pitch_motor_1_j
            pitch_motor_2_j = pitch_motor_2_i + delta_pitch_motor_2_j
    
            # now calculate y-offset
    
            delta_pitch_motor_1_tot = pitch_motor_1_j - pitch_motor_1_init
            delta_pitch_motor_2_tot = pitch_motor_2_j - pitch_motor_2_init
    
            delta_y_tot = -(delta_pitch_motor_1_tot + (delta_pitch_motor_2_tot - \
                                                     delta_pitch_motor_1_tot) * pitch_motor_ratio) * 0.3
            
            y_j = y_init + delta_y_tot
    
            # check we are within lims and populate list
            if pitch_motor_1_j <= pitch_motor_1_lims[0] or pitch_motor_1_j >= pitch_motor_1_lims[1] or\
               pitch_motor_2_j <= pitch_motor_2_lims[0] or pitch_motor_2_j >= pitch_motor_2_lims[1] or\
               y_j <= y_lims[0] or y_j >= y_lims[1]: 
                continue
                
            else:
                pitch_motor_1_steps.append(pitch_motor_1_j)
                pitch_motor_2_steps.append(pitch_motor_2_j)
                y_steps.append(y_j)

    return pitch_motor_1_steps, pitch_motor_2_steps, y_steps

class DelayLineAxis(PVPositioner):
    """Control for single axis of delay line"""
    tab_component_names = True
    
    setpoint = FCpt(EpicsSignal, '{prefix}:{_mtr_name}')
    readback = FCpt(EpicsSignalRO, '{prefix}:{_mtr_name}.RBV')
    done = FCpt(EpicsSignalRO, '{prefix}:{_mtr_name}.MOVN')
    done_value = 0 #0 when done
    egu='mm'
    fault = Cpt(EpicsSignalRO, ':FaultCalc')
    #faults
    fault_sensor = Cpt(EpicsSignalRO, ':SensorLoss', kind='omitted')
    fault_amplifer = Cpt(EpicsSignalRO, ':AmplifierFault', kind='omitted')
    fault_follow = Cpt(EpicsSignalRO, ':FatalFollowingError', kind='omitted')

    def __init__(self, prefix, motor_name=None, **kwargs):
        self._mtr_name = motor_name
        super().__init__(prefix, **kwargs)

class DelayLineDelay(PVPositionerDone):
    """Control the delay of the mirror."""
    tab_component_names = True
    
    setpoint = Cpt(EpicsSignal, ':MovAbs')
    readback = Cpt(EpicsSignalRO, ':Posn')
    us_gap = Cpt(EpicsSignalRO, ':US:Gap')
    ds_gap = Cpt(EpicsSignalRO, ':DS:Gap')
    fault = FCpt(EpicsSignalRO, 'TS:UNDS:MC01:AxisFault')
    delay = Cpt(EpicsSignal, ':TDES')

    
class SxrDelayLine(BaseInterface, Device):
    """
    Device to control SXR delay line axes individually.
    """
    tab_component_names = True
    
    tilt_x_pos = Cpt(DelayLineAxis, 'MOVR:UNDS:3567:POSX', motor_name='YTILTMOTR', kind='normal')
    tilt_x_neg = Cpt(DelayLineAxis, 'MOVR:UNDS:3567:NEGX', motor_name='YTILTMOTR', kind='normal')
    insertion_y = Cpt(DelayLineAxis, 'MOVR:UNDS:3555', motor_name='MOTR', kind='normal')
    mirror = Cpt(DelayLineDelay, 'MIRR:UNDS:3555', kind='normal')

sxr_delay_line_single = SxrDelayLine('', name='sxr_delay_line')

class SxrDelayLineCombo(SyncAxisTmo):
    """
    Device for controling SXR delay line. The default motion is a combined mover. You can also control each axis.
    """
    
    tab_component_names = True
    
    tilt_x_pos = Cpt(DelayLineAxis, 'MOVR:UNDS:3567:POSX', motor_name='YTILTMOTR', kind='normal')
    tilt_x_neg = Cpt(DelayLineAxis, 'MOVR:UNDS:3567:NEGX', motor_name='YTILTMOTR', kind='normal')
    insertion_y = Cpt(DelayLineAxis, 'MOVR:UNDS:3555', motor_name='MOTR', kind='normal')
    
#    pos = sync * scale + offset
    scales =  {'tilt_x_pos':1, 'tilt_x_neg':1, 'insertion_y':0.5}
    offsets = {'tilt_x_pos':-21.3166, 'tilt_x_neg':-19.201, 'insertion_y':-27.404}
    
    warn_deadband = 0.01
    fix_sync_keep_still = 'insertion_y'
    sync_limits = (-100, 100)
    
    @property
    def position(self):
        return super().position.sync

sxr_delay_line_positioner = SxrDelayLineCombo('', name='sxr_delay_line_positioner')

class SxrDelayLineFull(BaseInterface, Device):
    """
    Device with the combined motion of the positioner and the motion of the mirror separation (delay).
    """
    tab_component_names = True
    positioner = Cpt(SxrDelayLineCombo, '', kind='normal')
    mirror = Cpt(DelayLineDelay, 'MIRR:UNDS:3555', kind='normal')
    
sxr_delay_line = SxrDelayLineFull('', name='sxr_delay_line')    

