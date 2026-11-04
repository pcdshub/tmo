from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from ophyd.signal import EpicsSignal, EpicsSignalRO
from ophyd.device import Component as Cpt
from ophyd.device import FormattedComponent as FCpt
from ophyd.pv_positioner import PVPositioner
from pcdsdevices.interface import FltMvInterface
from ophyd.status import SubscriptionStatus
from tmo.tmo_beamline_control import unds
import numpy as np
import logging
logger = logging.getLogger(__name__)




class SXRSS_Chicane(FltMvInterface, PVPositioner):
    """
    Class for interacting with the soft x-ray self-seeding chicane as a motor. 
    In addition to the usual move methods, there are some convience functions:
        chicane delay which returns the current delay in fs (same as position)
        und_delay which estiatmes the delay from undulators after u36.
        estimate_delay which sums the two.
        bact_delay which estimates the delay based on the bend magnetic field (chicane_bend_act)
    """
    tab_component_names= True
    tab_whitelist = ['chicane_delay', 'und_delay', 'estimate_delay', 'bact_delay']
    
    setpoint = FCpt(EpicsSignal, 'RIX:USER:MCC:EPHOTK:SET2', kind = 'hinted')
    readback = FCpt(EpicsSignalRO, 'SIOC:SYS0:ML07:AO653', kind = 'hinted')
    done = FCpt(EpicsSignalRO,'SIOC:SYS0:ML07:AO652', kind='normal',
                doc='PV that is 0 while the motors are moving and 1 when device is ready for a new request.')
    done_value = 1
    chicane_bend_bact = Cpt(EpicsSignalRO, 'BEND:UNDS:3510:BACT', kind = 'omitted')
    
    def move(self, position, alert=True, wait=False, **kwargs):
        """
        Move to a specified position, optionally waiting for motion to complete, or write to screen.

        Parameters
        ----------
        position
            Position to move to
        alert
            flag to send an alert to the screen when the move completes, the default is True.
        moved_cb : callable
            Call this callback when movement has finished. This callback must accept one keyword argument: 'obj' which will be set to this positioner instance.
        timeout : float, optional
            Maximum time to wait for the motion. If None, the default timeout for this positioner is used.

        Returns
        -------
        status : MoveStatus

        Raises
        ------
        TimeoutError
            When motion takes longer than `timeout`
        ValueError
            On invalid positions
        RuntimeError
            If motion fails other than timing out

        """
        def log_the_move(obj):
            """send message when complete"""
            print('%s move complete' % obj.name)
            
        if alert:        
            st = super().move(position, moved_cb=log_the_move, wait=wait, **kwargs)
        else: 
            st = super().move(position, wait=wait, **kwargs)
        
        return st
    
    @property
    def chicane_delay(self):
        return self.readback.get()
    
    def und_delay(self, t0=0.140, del_und=None):
        delay = t0
        num_und=1
        kinit = unds.u36.k_act.get()
        u=37
        k=unds.u37.k_act.get()
        
        while ( (np.isclose(kinit, k, rtol=0.02)) and (u<47) ):
            num_und+=1
            u+=1
            uN = getattr(unds, 'u%s'%u)
            k = uN.k_act.get()
        
        if del_und is not None:
            delay += del_und * num_und
        else:
            delay += 0.125 * ((1+kinit**2/2)/4.7514) * num_und
        return delay
    
    @property
    def estimate_delay(self, **kwargs):        
        return self.chicane_delay + self.und_delay(**kwargs)
       
    @property
    def bact_delay(self, dispersion=2.221202E2):
        return dispersion * (self.chicane_bend_bact.get())**2
    
sxrss_chicane = SXRSS_Chicane(name='sxrss_chicane')