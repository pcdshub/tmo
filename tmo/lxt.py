from ophyd import Component as Cpt
from pcdsdevices.component import UnrelatedComponent as UCpt
from pcdsdevices.lxe import LaserTimingCompensation as _LaserTimingCompensation
from pcdsdevices.lxe import _ReversedTimeToolDelay
from pcdsdevices.epics_motor import SmarAct

txt = SmarAct('LM1K4:COM_MP2_DLY1', name='txt')

class _SmarActRevTimeToolDelay(_ReversedTimeToolDelay):
    motor = Cpt(SmarAct, '')

class LaserTimingCompensation(_LaserTimingCompensation):
    delay = UCpt(_SmarActRevTimeToolDelay, doc='The **reversed** txt motor')

lxt_ttc = LaserTimingCompensation('', delay_prefix='LM1K4:COM_MP2_DLY1', laser_prefix='LAS:FS14', name='lxt_ttc')
lxt_ttc.delay.n_bounces = 16

lxt = lxt_ttc.laser

#import logging
#logging.getLogger('pint').setLevel(logging.ERROR)
