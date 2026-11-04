from tmo.db import im2k0, im1k4, im2k4, im3k4, im4k4, im5k4, im6k4, tm1k4, tm2k4, pf1k4, pf2k4
from tmo.db import sl1k0, sl2k0, sl1k4
from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from ophyd.device import Component as Cpt
from ophyd.device import FormattedComponent as FCpt
from pcdsdevices.device import ObjectComponent as OCpt
from ophyd.signal import EpicsSignal, EpicsSignalRO, Signal
import pandas as pd
from collections import OrderedDict
from tmo.tmo_utils import log_positions as _log_positions
from tmo.tmo_utils import add_tags as _add_tags
from tmo.tmo_utils import EpicsSignalPutLog
import logging
logger = logging.getLogger(__name__)

class PPM_Alignment(BaseInterface, Device):
    """
    Class for interacting with PPM alignment PVs. 
    The position of the beam on the PPM_screen is given by the _act attribute. 
    The target is given by the _set attribute.
    _diff gives the difference between the _act and the _set
    This difference can be logged with a screen shot using log_diff 
    """
    tab_component_names = True
    tab_whitelist = ['x_diff','y_diff', 'log_diff']
    x_act = Cpt(EpicsSignalRO, ':PPM:CAM:X_BM_CTR', kind='normal')
    x_set = Cpt(EpicsSignalPutLog, ':PPM:CAM:X_RTCL_CTR', kind='normal')
    y_act = Cpt(EpicsSignalRO, ':PPM:CAM:Y_BM_CTR', kind='normal')
    y_set = Cpt(EpicsSignalPutLog, ':PPM:CAM:Y_RTCL_CTR', kind='normal')
    target_state = Cpt(EpicsSignalRO, ':PPM:MMS:STATE:GET_RBV', string=True, kind='omitted')
    transmission = FCpt(EpicsSignalRO, 'AT1K0:GAS:TRANS_RBV', kind='omitted')
    photon_energy = FCpt(EpicsSignalRO, 'SIOC:SYS0:ML00:AO628', kind='omitted')
    und_47_x = FCpt(EpicsSignalRO, 'BPMS:UNDS:4790:XOFF.D', kind='omitted' )
    und_47_y = FCpt(EpicsSignalRO, 'BPMS:UNDS:4790:YOFF.D', kind='omitted' )
    und_34_x = FCpt(EpicsSignalRO, 'BPMS:UNDS:3490:XOFF.D', kind='omitted' )
    und_34_y = FCpt(EpicsSignalRO, 'BPMS:UNDS:3490:YOFF.D', kind='omitted' )
    
    def __init__(self, prefix, **kwargs):
        self.cam = prefix.lower()
        return super().__init__(prefix, **kwargs) 
        
    @property
    def x_diff(self):
        return self.x_act.get() - self.x_set.get()
    
    @property
    def y_diff(self):
        return  self.y_act.get() - self.y_set.get()
    
    def log_diff(self, log=False, msg=None, **kwargs):
        diff = OrderedDict()
        diff ['imager'] = self.cam
        diff ['x diff'] = self.x_diff
        diff ['y diff'] = self.y_diff
        diff ['target'] = self.target_state.get()
        diff [self.transmission.pvname] = self.transmission.get()
        diff ['photon energy'] = self.photon_energy.get()
        diff [self.und_34_x.pvname] = self.und_34_x.get()
        diff [self.und_34_y.pvname] = self.und_34_y.get()
        diff [self.und_47_x.pvname] = self.und_47_x.get()
        diff [self.und_47_y.pvname] = self.und_47_y.get()
        #also get slit positions
        for slit in [sl1k0,sl2k0,sl1k4]:
            for axis in ['vo','vg','ho','hg']:
                dev = getattr(slit, axis)
                diff[dev.name] = dev.wm()
        
        kwargs['tags'] = _add_tags(kwargs.get('tags', ''),('%s alignment' % (self.cam) ) )  
        _log_positions(diff, log=log, msg=msg, include_cam=self.cam, **kwargs)
        

class TMO_PPM(BaseInterface, Device):
    """
    Class for interacting with the PPM_screen PVs for all TMO Imagers
    """
    tab_component_names = True
    tab_whitelist = []
    im2k0_ppm = Cpt(PPM_Alignment, 'IM2K0', kind='normal')
    im2k4_ppm = Cpt(PPM_Alignment, 'IM2K4', kind='normal')
    im3k4_ppm = Cpt(PPM_Alignment, 'IM3K4', kind='normal')
    im4k4_ppm = Cpt(PPM_Alignment, 'IM4K4', kind='normal')
    im5k4_ppm = Cpt(PPM_Alignment, 'IM5K4', kind='normal')
    im6k4_ppm = Cpt(PPM_Alignment, 'IM6K4', kind='normal')    

class Tmo_Alignment(BaseInterface, Device):
    tab_component_names = True
    tab_whitelist = ['summary', 'all_out', 'next_imager', 'prev_imager', 'imager_out','imager_in']
    
    current_imager = Cpt(Signal, kind='normal')
    im2k0 = OCpt(im2k0, kind='normal')
    im1k4 = OCpt(im1k4, kind='normal')
    im2k4 = OCpt(im2k4, kind='normal')
    im3k4 = OCpt(im3k4, kind='normal')
    im4k4 = OCpt(im4k4, kind='normal')
    im5k4 = OCpt(im5k4, kind='normal')
    im6k4 = OCpt(im6k4, kind='normal')
    tm1k4 = OCpt(tm1k4, kind='normal')
    tm2k4 = OCpt(tm2k4, kind='normal')
    pf1k4 = OCpt(pf1k4, kind='normal')
    pf2k4 = OCpt(pf2k4, kind='normal')
    PPM = Cpt(TMO_PPM, '')
    
    def __init__(self, prefix, curr_im=1, **kwargs):    
        self.imList = ['im2k0', 'im1k4', 'im2k4', 'im3k4', 'im4k4', 'im5k4', 'im6k4', 'tm1k4', 'tm2k4']
#        self.current_imager = curr_im
        return super().__init__(prefix, **kwargs)
        
    def _move_to_target(self, imager_id):
        imager = getattr(self, self.imList[imager_id])
        if ('PPM' in type(imager).__name__) :
            imager.target("YAG2")     
        if ('XPIM' in type(imager).__name__) :
            imager.target("YAG")
        if ('TM' in type(imager).__name__):
            imager.target("YAG")
            
    def _velocity(self, imager, vel=None):
        if type(imager) == int:
            imager = getattr(self, self.imList[imager])        
        if hasattr(imager, 'target'):
            num_state = imager.target.config.state_count
            config = imager.target.config
        elif hasattr(imager, 'config'):
            num_state = imager.config.state_count
            config = imager.config
        else:
            logger.error('%s does not have a "target" attribute or a "config" attribute.' % (imager.name))
            return
        
        old_vel_dict=OrderedDict()
        new_vel_dict=OrderedDict()
        for n in range(num_state):
            state = getattr(config, ('state0%d' % (n+1)))
            state_name = state.state_name.get()
            old_vel_dict[state_name] = state.velo.get()
            if vel is not None:
                state.velo.put(vel)
                new_vel_dict[state_name] = vel
        if vel is None:
            _log_positions(old_vel_dict, log=False)
        else:
            _log_positions(old_vel_dict, log=True, msg='Update state velocities', set_points=new_vel_dict, tags='update velocity')
                
        
    def imager_out(self, imager_id=-1):
        if (imager_id > -1):
            self.current_imager.put(imager_id)
        imager = getattr(self, self.imList[self.current_imager.get()])
        imager.target("OUT")
        if ('PPM' in type(imager).__name__) :
            imager.led.set(0)
    
    def next_imager(self):
        self.imager_out(self.current_imager.get())
        self.current_imager.put(self.current_imager.get() + 1)
        self._move_to_target(self.current_imager.get())
            
    def prev_imager(self, remove_curr=True):
        if remove_curr:
            self.imager_out(self.current_imager.get())
        self.current_imager.put(self.current_imager.get() - 1)
        self._move_to_target(self.current_imager.get())

    def imager_checkout(self, curr_imager):
        self.current_imager.put(curr_imager)
        self._move_to_target(self.current_imager.get())
        imager = self.imList[self.current_imager.get()]
        if hasattr(imager, 'led'):
            imager.led.set(5.6)
    
    def imager_in(self, imager, remove_curr=True):
        if remove_curr:
            self.imager_out(self.current_imager.get())
        self.current_imager.put(imager)
        self._move_to_target(self.current_imager.get())
    
    @property
    def summary(self):
        summary_dict=OrderedDict()
        for ind, im in enumerate(self.imList):
            imager = getattr(self, im)
            summary_dict[imager.name]=imager.target()
        _log_positions(summary_dict, log=False)
        
    
    def all_out(self):
        for ind, im in enumerate(self.imList):
            imager = getattr(self, im)
            imager.target("OUT")
        pf1k4.target("OUT")
        pf2k4.target("OUT")
        
tmo_imagers = Tmo_Alignment('',name='tmo_imagers')

    
    
    