from ophyd.device import Component as Cpt
from ophyd import EpicsSignal
from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from collections import OrderedDict
from pcdsdevices.tpr import TprMotor, TprTrigger, TimingMode

from tmo.db import elog
import pandas as pd
import time

class Tpr(BaseInterface, Device):
    tab_component_names = True
    tab_whitelist = ['clear_faults_all', 'status']
    
    tRef = Cpt(EpicsSignal, ':MSGDLY', kind='config')
    ch00 = Cpt(TprTrigger, '', channel=0, timing_mode=TimingMode.LCLS2)
    ch01 = Cpt(TprTrigger, '', channel=1, timing_mode=TimingMode.LCLS2)
    ch02 = Cpt(TprTrigger, '', channel=2, timing_mode=TimingMode.LCLS2)
    ch03 = Cpt(TprTrigger, '', channel=3, timing_mode=TimingMode.LCLS2)
    ch04 = Cpt(TprTrigger, '', channel=4, timing_mode=TimingMode.LCLS2)
    ch05 = Cpt(TprTrigger, '', channel=5, timing_mode=TimingMode.LCLS2)
    ch06 = Cpt(TprTrigger, '', channel=6, timing_mode=TimingMode.LCLS2)
    ch07 = Cpt(TprTrigger, '', channel=7, timing_mode=TimingMode.LCLS2)
    ch08 = Cpt(TprTrigger, '', channel=8, timing_mode=TimingMode.LCLS2)
    ch09 = Cpt(TprTrigger, '', channel=9, timing_mode=TimingMode.LCLS2)
    ch10 = Cpt(TprTrigger, '', channel=10, timing_mode=TimingMode.LCLS2)
    ch11 = Cpt(TprTrigger, '', channel=11, timing_mode=TimingMode.LCLS2)
    
tmo_misc_tpr = Tpr('TMO:MISC:TPR:01', name='tmo_misc_tpr')