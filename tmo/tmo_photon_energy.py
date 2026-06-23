from pcdsdevices.beam_stats import BeamEnergyRequest, BeamEnergyRequestNoWait
from pcdsdevices.pseudopos import SyncAxis
from pcdsdevices.device import ObjectComponent as OCpt
from ophyd.device import FormattedComponent as FCpt
from ophyd.device import Component as Cpt
from ophyd.signal import EpicsSignal, EpicsSignalRO, Signal
from ophyd.status import WaitTimeoutError, SubscriptionStatus, DeviceStatus
from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from tmo.tmo_utils import log_positions as _log_positions
from tmo.tmo_utils import EpicsSignalROStop
import time
import logging
logger = logging.getLogger(__name__)


hf_w = BeamEnergyRequest('TMO',name='hf_w',atol=0.05,skip_small_moves=False,pv_index=1)
hf_2w = BeamEnergyRequest('TMO',name='hf_2w',atol=0.05,skip_small_moves=False,pv_index=2)

hf_w_status = BeamEnergyRequest('TMO',name='hf_w',pv_index=1,acr_status_suffix='AO801')
hf_2w_status = BeamEnergyRequest('TMO',name='hf_2w',pv_index=2,acr_status_suffix='AO802')

state_enum ={
    -7: 'error state',
    -6: 'listening stopped',
    -5: 'listening stopped by self-destruction',
    -4: 'setpoint invalid (contains NaN)',
    -3: 'difference between K and Vernier is too large',
    -2: 'setpoint out of range: too low',
    -1: 'setpoint out of range: too high',
     0: 'no new values in setpoint, ready for command',
     1: 'new value is ok, moving'
     }

class BeamEnergyRequestCheckNew(BeamEnergyRequestNoWait):
    tab_component_names = True
    tab_whitelist = ['pvname','state']
    
    state_pv = FCpt(
        EpicsSignalROStop,
        'SIOC:SYS0:ML07:{acr_state_suffix}',
        kind='normal',
        add_prefix=('suffix', 'write_pv', 'acr_state_suffix'),
        doc=(
            'PV to indicate status of request'
        )
    )
    
    target = FCpt(
        EpicsSignalRO,
        'SIOC:SYS0:ML07:{acr_target_suffix}',
        kind='normal',
        add_prefix=('suffix', 'write_pv', 'acr_target_suffix'),
        doc=(
            'Current target postions used to check if motor is ready to move.'
        )
    )
    
    done_moving = FCpt(
        EpicsSignalRO,
        'SIOC:SYS0:ML07:{acr_move_status_suffix}',
        kind='normal',
        add_prefix=('suffix', 'write_pv', 'acr_status_suffix'),
        doc=(
            'PV that is 0 while the motors are moving and 1 when ACR is '
            'ready for a new request. ACR can pick which of these PVs '
            'to use to report status.'
        )
    )
       

    last_error = Cpt(Signal, value=None, kind='omitted', doc='for tracking the last error during subscription.')
    
    def move(self, position, **kwargs):
        
        def check_move(*, old_value, value, **kwargs):
            "Wait for state PV to change and look for errors."
            logger.debug('old_value: %s, value: %s' % (old_value, value))
            if value < 0:
                self.last_error.put(value)
                logger.error('Error: %s'%(state_enum[value]))
            return (value > 0)
            
        def wait_for_ready(*, old_value, value, **kwargs):
            "Wait for state PV to reach ready state (0)"
            logger.debug('old_value: %s, value: %s' % (old_value, value))
            return (value==0)
        
        def reset_setpoint(status):
            "Reset the sepoint PV to the last accepted value if move fails."
            error = status.exception()
            if error is None:
                logger.debug('undulator moving to %s, last accepted is now: %s' % (self.setpoint.get(), self.target.get()))
            else:
                logger.debug('Return setpoint to last accepted (%s)' % (self.target.get()) )
                if self.target.get() > 0:
                    self.setpoint.put(self.target.get())
                else:
                    logger.error('No previous value in last accepted.')
#                status.set_exception(Exception('Error: %s' % state_enum[value]))
#                raise Exception('Error: %s' % state_enum[value])
            
        initial_setpoint = self.setpoint.get()
        timeout = kwargs.get('timeout', 10)
        timeout = timeout if timeout is not None else 10
        
        logger.debug('Move requested. current setpoint: %s, requested position: %s, last accepted position: %s, done moving: %s'
                     % (initial_setpoint, position, self.target.get(), self.done_moving.get()))
        
        state_value = self.state_pv.get()
        logger.debug('State PV gives %s: %s' %(state_value, state_enum[state_value]))
        
        if state_value < -4:
            logger.debug('State PV indicated a bad state %s' % self.state_pv.get())
            self.last_error.put(state_value)
            raise Exception('Error: %s' % state_enum[self.state_pv.get()])
            return DeviceStatus(self, done=True, success=False)
        
        elif state_value == 1:
            logger.debug('waiting for state PV to reach ready state.')
            status = SubscriptionStatus(self.state_pv, wait_for_ready, timeout=timeout)
            status.wait()
            if not status.success:                    
                raise WaitTimeoutError("State PV failed to reach ready state before timeout (%s sec)." % (timeout))        
                return status
                           
        logger.debug('State PV indicates ready for move, moving to %s' % position)        
        result = super().move(position, **kwargs)
        
        if position == initial_setpoint:
            return DeviceStatus(self, done=True, success=True)
        
        logger.debug('waiting for state PV to update.')
        status = SubscriptionStatus(self.state_pv, check_move, timeout=timeout)
        status.add_callback(reset_setpoint)                       
        return status
           
    @property
    def pvname(self):
        pv_dict = {
                 'setpoint': self.setpoint.pvname,
                      'ref': self.ref.pvname,
            'last accepted': self.target.pvname,
                   'status': self.done_moving.pvname,
                 'state_pv': self.state_pv.pvname
            
            }
        _log_positions(pv_dict, log=False)
        
    @property
    def state(self):
        return state_enum[self.state_pv.get()]
    
    def __init__(self, acr_target_suffix=None, acr_move_status_suffix=None, acr_state_suffix=None, *args, **kwargs):
        self.acr_state_suffix = acr_state_suffix
        self.acr_target_suffix = acr_target_suffix
        self.acr_move_status_suffix = acr_move_status_suffix
        super().__init__(*args, atol=0.1, skip_small_moves=False, **kwargs)
        if self.setpoint.get() < 0:
            logger.info('Found -1 in PV setpoint.')#', replacing with ref.')
            #self.setpoint.put(self.ref.get())
            

hf_w_test = BeamEnergyRequestCheckNew( prefix='TMO', name='hf_w', pv_index=1, acr_target_suffix='AO821', acr_move_status_suffix='AO801', acr_state_suffix='AO916')

class BeamEnergyRequestCheck(BeamEnergyRequestNoWait):
    tab_component_names = True
    tab_whitelist = ['pvname','state']
    
    state_pv = FCpt(
        EpicsSignal,
        'SIOC:SYS0:ML07:{acr_state_suffix}',
        kind='normal',
        add_prefix=('suffix', 'write_pv', 'acr_state_suffix'),
        doc=(
            'Current target postions used to check if motor is ready to move.'
        )
    )
    
    target = FCpt(
        EpicsSignal,
        'SIOC:SYS0:ML07:{acr_target_suffix}',
        kind='normal',
        add_prefix=('suffix', 'write_pv', 'acr_target_suffix'),
        doc=(
            'Current target postions used to check if motor is ready to move.'
        )
    )
    
    done_moving = FCpt(
        EpicsSignal,
        'SIOC:SYS0:ML07:{acr_move_status_suffix}',
        kind='normal',
        add_prefix=('suffix', 'write_pv', 'acr_status_suffix'),
        doc=(
            'PV that is 0 while the motors are moving and 1 when ACR is '
            'ready for a new request. ACR can pick which of these PVs '
            'to use to report status.'
        )
    )
    
#    def _setup_move(self, position):
#        if abs(self.position-self.target.get()) > 0.1:
#            logger.debug('Motor has not reached target and a new move was issued')
#            while abs(self.position-self.target.get()) > 0.1:
#                time.sleep(0.1)
#        super()._setup_move(position)    
#        return
    
    def _check_move(self, position, timeout):
        if abs(position - self.target.get()) > 0.1:
            logger.debug('Last accepted (%s), is not updated to requested %s.' % (self.target.get(), position))
            if self.acr_move_status_suffix is None:
                logger.debug('No move status PV specified, checking target position')
                while abs(position - self.target.get()) > 0.1:
                    self.setpoint.put(position)
                    time.sleep(0.1)
                logger.debug('Setpoint set to %s' % position)
            else:
                logger.debug('Monitoring move status to wait for previous move to complete, current status:%s' % self.done_moving.get())
                timer = 0
                while not self.done_moving.get():
                    time.sleep(0.05)
                    timer += 0.05
                    if timer > timeout:
                        self.setpoint.put(self.target.get())
                        raise WaitTimeoutError("Previous move failed to complete before timeout (%s sec)." % (timeout))        
                logger.debug('Previous move complete moving to %s' % position)
                self.setpoint.put(position)
#        w_NoCheck = Cpt(BeamEnergyRequest, 'TMO', atol=0.1, skip_small_moves=False, pv_index=1, kind='omitted')

    def move(self, position, **kwargs):
              
        initial_setpoint = self.setpoint.get()
        timeout = kwargs.get('timeout', 10)
        timeout = timeout if timeout is not None else 10
        
        logger.debug('current setpoint: %s, requested position: %s, last accepted position: %s, done moving: %s'
                     % (initial_setpoint, position, self.target.get(), self.done_moving.get()))
        result = super().move(position, **kwargs)
        
        if abs(initial_setpoint - position) > 0.1:
            logger.debug('waiting for move to start')
            timer = 0
            while self.done_moving.get(): 
                time.sleep(0.01)
                timer += 0.01
                if timer > timeout:
                    self.setpoint.put(initial_setpoint)
                    raise WaitTimeoutError("Move failed to start before timeout (%s sec). \nIs the %s PV configured?" % (timeout,self.setpoint.pvname))        
            logger.debug('move has started.')
        self._check_move(position, timeout)
        return result
    
    @property
    def pvname(self):
        pv_dict = {
                 'setpoint': self.setpoint.pvname,
                      'ref': self.ref.pvname,
            'last accepted': self.target.pvname,
                   'status': self.done_moving.pvname,
                 'state_pv': self.state_pv.pvname
            
            }
        _log_positions(pv_dict, log=False)
        
    @property
    def state(self):
        return state_enum[self.state_pv.get()]
    
    def __init__(self, acr_target_suffix=None, acr_move_status_suffix=None, acr_state_suffix=None, *args, **kwargs):
        self.acr_state_suffix = acr_state_suffix
        self.acr_target_suffix = acr_target_suffix
        self.acr_move_status_suffix = acr_move_status_suffix
        super().__init__(*args, atol=0.1, skip_small_moves=False, **kwargs)

hf_w_check = BeamEnergyRequestCheckNew( prefix='TMO', name='hf_w', pv_index=1, acr_target_suffix='AO821', acr_move_status_suffix='AO801', acr_state_suffix='AO916')
hf_2w_check = BeamEnergyRequestCheckNew( prefix='TMO', name='hf_2w', pv_index=2, acr_target_suffix='AO823', acr_move_status_suffix='AO802', acr_state_suffix='AO917')
    
class HF_W2W(SyncAxis):
    hf_w = OCpt(hf_w_check)
    hf_2w = OCpt(hf_2w_check)
    tab_component_names = True
    scales = {'hf_w': 0.5, 'hf_2w':1}
    warn_deadband = 1E-2
    fix_sync_keep_still = 'hf_2w'
    sync_limits = (350, 2E3)

hf_w2w = HF_W2W('', name='hf_w2w')

class HF_W3W(SyncAxis):
    hf_w = OCpt(hf_w_check)
    hf_2w = OCpt(hf_2w_check)
    tab_component_names = True
    scales = {'hf_w': (1/3),'hf_2w':1}
    warn_deadband = 1E-2
    fix_sync_keep_still = 'hf_2w'
    sync_limits = (350, 2E3)

hf_w3w = HF_W3W('', name='hf_w3w')

class HF(BaseInterface, Device):
    """
    Photon Energy control PVs for TMO. 
    Attributes:
        w: Control of set1 PVs: TMO:USER:MCC:EPHOTK:SET1
        w2: Control of set2 PVs: TMO:USER:MCC:EPHOTK:SET2
        w2w: Combined mover for w/2w operation. The setpoint is controlled by set2 PV (TMO:USER:MCC:EPHOTK:SET2), set 1 PV is moved by half the requested amount.
        w_status: set1 PV control with movement readback from SIOC:SYS0:ML07:AO801
        w2_status: set2 PV control with movement readback from SIOC:SYS0:ML07:AO802
    """
    tab_component_names = True
       
    w = Cpt(BeamEnergyRequestCheckNew, prefix='TMO', pv_index=1, acr_target_suffix='AO821', acr_move_status_suffix='AO801',acr_state_suffix='AO916')
#    w = Cpt(BeamEnergyRequest, 'TMO', atol=0.1, skip_small_moves=False, pv_index=1)
    w2 = Cpt(BeamEnergyRequestCheckNew, prefix='TMO', pv_index=2, acr_target_suffix='AO823', acr_move_status_suffix='AO802',acr_state_suffix='AO917')
#    w2 = Cpt(BeamEnergyRequest, 'TMO', atol=0.1, skip_small_moves=False, pv_index=2)
    w2w = Cpt(HF_W2W, '')
    w_status = Cpt(BeamEnergyRequest, 'TMO', pv_index=1, acr_status_suffix='AO801')
    w2_status = Cpt(BeamEnergyRequest, 'TMO', pv_index=2, acr_status_suffix='AO802')
    w_NoCheck = Cpt(BeamEnergyRequest, 'TMO', atol=0.1, skip_small_moves=False, pv_index=1, kind='omitted')
    w2_NoCheck = Cpt(BeamEnergyRequest, 'TMO', atol=0.1, skip_small_moves=False, pv_index=2, kind='omitted')     
hf = HF(name='hf')

"""
class HF2(BaseInterface, Device):
    tab_component_names = True
       
    w = Cpt(BeamEnergyRequest, 'TMO', atol=0, skip_small_moves=False, pv_index=1)
    w2 = Cpt(BeamEnergyRequest, 'TMO', atol=0, skip_small_moves=False, pv_index=2)
    w2w = Cpt(SyncAxis, 
              w=Cpt(BeamEnergyRequest, 'TMO', atol=0, skip_small_moves=False, pv_index=1), 
              w2=Cpt(BeamEnergyRequest, 'TMO', atol=0, skip_small_moves=False, pv_index=2), 
              fix_sync_keep_still = 'w2', scales = {'w': 0.5}, 
              tab_component_names = True, warn_deadband = 1E-2, sync_limits = (350, 2E3))
    w_status = Cpt(BeamEnergyRequest, 'TMO', pv_index=1, acr_status_suffix='AO801')
    w2_status = Cpt(BeamEnergyRequest, 'TMO', pv_index=2, acr_status_suffix='AO802')
    
#    def __init__(self, prefix='', name='hf', **kwargs):
#        super().__init__(prefix=prefix, name=name, **kwargs)
#        self.w2w.w = self.w
#        self.w2w.w2 = self.w2
hf2 = HF2(name='hf2')
"""