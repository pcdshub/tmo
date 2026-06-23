from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from pcdsdevices.epics_motor import SmarAct, SmarActOpenLoop
from pcdsdevices.pseudopos import SyncAxis, delay_instance_factory
from pcdsdevices.lasers.elliptec import EllRotation
from ophyd.device import Component as Cpt
from pcdsdevices.device import ObjectComponent as OCpt
from tmo.tmo_utils import log_positions as _log_positions
#from tmo.tmo_laser import SmarActThreeAxis

llt = delay_instance_factory('LM1K4:EJX_MSC_DLY', motor_class=SmarAct, egu='s', n_bounces=2, name='llt')

class UserTarget(BaseInterface, Device):
    tab_component_names = True
    
    x = Cpt(SmarAct, ':m2', kind='normal', doc='MSC sample paddle motion, horizontal')
    y = Cpt(SmarAct, ':m1', kind='normal', doc='MSC sample paddle motion, vertical')
    r = Cpt(SmarAct, ':m3', kind='normal', doc='MSC sample paddle motion, rotation')   

class SmarActThreeAxis(BaseInterface, Device):
    
    tab_component_names=True
    x = Cpt(SmarAct, '_LS2', kind='normal', 
                 doc='optic motion x')
    y = Cpt(SmarAct, '_LS3', kind='normal', 
                 doc='optic motion y')
    z = Cpt(SmarAct, '_LS1', kind='normal', 
                 doc='optic motion z')
    
class User(BaseInterface, Device):
    
    tab_component_names= True
    tab_whitelist = ['get_user']#, 'llt', 'lxt_stc']
    
    mrco_lens = Cpt(SmarActThreeAxis, 'LM1K4:EJX_MRCO')
    msc_lens = Cpt(SmarActThreeAxis, 'LM1K4:EJX_MSC')
    mrco_qwp = Cpt(EllRotation, 'LM1K4:EJX:ELL:2', channel=1, port=0, kind='normal', atol = 0.05)
    msc_hwp = Cpt(EllRotation, 'LM1K4:EJX:ELL:2', channel=2, port=0, kind='normal', atol = 0.05)
    mrco_ap = Cpt(SmarActOpenLoop, 'LM1K4:EJX_MRCO_AP', kind='normal')  
    llt = OCpt(llt)
    sample = Cpt(UserTarget, 'TMO:MCS2:02', kind='normal', doc='sample paddle motion.')   

    def __init__(self):
        name = 'user'
        return super().__init__(name=name)

    def get_user(self, log=False, msg=None, **kwargs):
        curr_positions= {self.mrco_qwp.name: self.mrco_qwp.position,
                          self.msc_hwp.name: self.msc_hwp.position,
                          self.mrco_lens.x.name: self.mrco_lens.x.position,
                          self.mrco_lens.y.name: self.mrco_lens.y.position,
                          self.mrco_lens.z.name: self.mrco_lens.z.position,
                          self.msc_lens.x.name: self.msc_lens.x.position,
                          self.msc_lens.y.name: self.msc_lens.y.position,
                          self.msc_lens.z.name: self.msc_lens.z.position,
                          self.llt.name: self.llt.delay.position,
                          self.llt.motor.name: self.llt.motor.position,
                          self.llt.user_offset.name: self.llt.user_offset.get(),
                          self.sample.x.name: self.sample.x.position,
                          self.sample.y.name: self.sample.y.position,
                          self.sample.r.name: self.sample.r.position,
                          }
        kwargs['tags'] = kwargs.get('tags', '') + ' user_device'                            
        _log_positions(curr_positions, log, msg=msg, **kwargs) 


