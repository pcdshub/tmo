#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from tmo.tmo_mpods import get_mpod
from tmo.tmo_beamline_control import get_lamp_sq1, get_lamp_motion
from tmo.tmo_optics import get_fzp
from tmo.db import sp1k4
from tmo.tmo_beamline_control import _log_positions
import sys
sys.path.append('/cds/home/opr/tmoopr')

class User():
    def log_run_parameters(self,run_num, msg=None):
        """
        Log the setting of various devices: MPOD, LAMP motion, FZP
    
        Parameters
        ----------
        run_num : int
            Run number to tag.
        msg : str, optional
            Message to add to the elog post. The default is None.
    
        Returns
        -------
        None.
    
        """
        get_mpod(log=True, msg=msg, run=run_num)
        get_lamp_motion(log=True, msg=msg, run=run_num)
        get_fzp(log=True, msg=msg, run=run_num)
        
    def get_fzp_states(self, state_num):
        state_str = ('_state%02d' % state_num)
        state_dict={'state_name': getattr(sp1k4.zone_plate.config,('m1'+state_str)).state_name.get(),
                    'x':getattr(sp1k4.zone_plate.config,('m1'+state_str)).setpoint.get(),
                    'y':getattr(sp1k4.zone_plate.config,('m2'+state_str)).setpoint.get(),
                    'z':getattr(sp1k4.zone_plate.config,('m3'+state_str)).setpoint.get()}
        _log_positions(state_dict, log=False)
