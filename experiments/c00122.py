#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from tmo.tmo_mpods import get_mpod
from tmo.tmo_optics import get_fzp
from tmo.db import sp1k4
from tmo.tmo_beamline_control import _log_positions
import numpy as np
import sys
sys.path.append('/cds/home/opr/tmoopr')

class User():
    
        
    def get_fzp_states(self, state_num):
        state_str = ('_state%02d' % state_num)
        state_dict={'state_name': getattr(sp1k4.zone_plate.config,('m1'+state_str)).state_name.get(),
                    'x':getattr(sp1k4.zone_plate.config,('m1'+state_str)).setpoint.get(),
                    'y':getattr(sp1k4.zone_plate.config,('m2'+state_str)).setpoint.get(),
                    'z':getattr(sp1k4.zone_plate.config,('m3'+state_str)).setpoint.get()}
        _log_positions(state_dict, log=False)


#tmo_lamp_sq1 = SQR1('MCQ:TMO:01', name='tmo_lamp_sq1')
