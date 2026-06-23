from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from pcdsdevices.epics_motor import SmarAct
from pcdsdevices.pseudopos import SyncAxis, delay_instance_factory
from ophyd.device import Component as Cpt
from pcdsdevices.device import ObjectComponent as OCpt
from tmo.tmo_utils import log_positions as _log_positions
from tmo.lxt2 import lxt

llt = delay_instance_factory('LM1K4:EJX_TP1_DLY1', motor_class=SmarAct, egu='s', n_bounces=2, name='llt')

class LXTSTC(SyncAxis):
    lxt = OCpt(lxt)
    llt = OCpt(llt)
    tab_component_names = True
    scales = {'llt': -1}
    warn_deadband = 5e-14
    fix_sync_keep_still = 'lxt'
    sync_limits = (-10e-6, 10e-6)

class UserTarget(BaseInterface, Device):
    tab_component_names = True
    
    x = Cpt(SmarAct, ':m2', kind='normal', doc='MSC sample paddle motion, horizontal')
    y = Cpt(SmarAct, ':m1', kind='normal', doc='MSC sample paddle motion, vertical')
    r = Cpt(SmarAct, ':m3', kind='normal', doc='MSC sample paddle motion, rotation')   

class UserLens(BaseInterface, Device):
    tab_component_names = True
    
    x = Cpt(SmarAct, '_LM2', kind='normal', doc='lens motion x')
    y = Cpt(SmarAct, '_LM3', kind='normal', doc='lens motion y')
    z = Cpt(SmarAct, '_LM1', kind='normal', doc='lens motion z')
    
    
class User(BaseInterface, Device):
    
    tab_component_names= True
    tab_whitelist = ['get_user', 'llt', 'lxt_stc']
    
    lens_x = Cpt(SmarAct, 'LM1K4:EJX_TP1_LS1_LM2', kind='normal', doc='div lens motion x')
    lens_y = Cpt(SmarAct, 'LM1K4:EJX_TP1_LS1_LM3', kind='normal', doc='div lens motion y')
    lens_z = Cpt(SmarAct, 'LM1K4:EJX_TP1_LS1_LM1', kind='normal', doc='div lens motion z')
    
    sample = Cpt(UserTarget, 'TMO:MCS2:02', kind='normal', doc='sample paddle motion.')
    lens = Cpt(UserLens, 'LM1K4:EJX_TP1_LS1', kind='omitted', doc='diverging lens.')    

    def __init__(self):
        name = 'user'
        self.llt = llt
        self.lxt_stc = LXTSTC('', name='lxt_stc')
        return super().__init__(name=name)

    def get_user(self, log=False, msg=None, **kwargs):
        curr_positions= {self.lens.x.name: self.lens.x.position,
                         self.lens.y.name: self.lens.y.position,
                         self.lens.z.name: self.lens.z.position,
                         self.sample.x.name: self.sample.x.position,
                         self.sample.y.name: self.sample.y.position,
                         self.sample.r.name: self.sample.r.position,
                         self.llt.name: self.llt.delay.position,
                         self.llt.motor.name: self.llt.motor.position,
                         self.llt.user_offset.name: self.llt.user_offset.get()
                         }
        kwargs['tags'] = kwargs.get('tags', '') + ' user_device'                            
        _log_positions(curr_positions, log, msg=msg, **kwargs) 
        