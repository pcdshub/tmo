import numpy as np
from pcdsdevices.device import ObjectComponent as OCpt
from pcdsdevices.pseudopos import SyncAxis
from tmo.db import mr2k4_kbo, mr3k4_kbo, mr1k4_soms, sp1k4, mr4k4_kbo, mr5k4_kbo
from epics import caget, caput, cainfo
from elog import HutchELog as elog_temp
elog = elog_temp.from_registry()
import pandas as pd


def fzp_checkout():

    for motor in dir(sp1k4):
        if (motor != 'get' or motor != 'stop'):
            temp = getattr(sp1k4,motor)
            try:
                temp.umvr(0.01, timeout=1, newline=True)
            except BaseException:
                print(motor, "failed")
                return False
            else:
                temp.umvr(-0.01, timeout=1)
                print(motor, "checked out")
    print("FZP motion checkout complete")
    return True             
            
