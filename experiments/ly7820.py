import numpy as np
from tmo.db import im2k4, tmo_lamp
from elog import HutchELog as elog_temp
elog = elog_temp.from_registry()
from tmo.end_beamTime import transmission_set
from epics import caget, caput, cainfo
import pandas as pd

yag_positions = {"magnet_x": -8.207, "magnet_y": 21.179, "magnet_z": -2, "needle_x": 3.305, "needle_y": 90, "needle_z": 2.5, "sq1_x": 29.0611, "sq1_y": -3.60, "sq1_z": 7.99, "sq1_rx": -0.279, "sq1_ry": -11.005, "sq1_rz": -6.55, "laser_wp":-35}
siN_positions = {"magnet_x": -8.104, "magnet_y": 21.818, "magnet_z": -2, "needle_x": 3.305, "needle_y": 90, "needle_z": 2.5, "sq1_x": 29.0611, "sq1_y": -3.60, "sq1_z": 7.99, "sq1_rx": -0.279, "sq1_ry": -11.005, "sq1_rz": -6.55, "laser_wp":-35}

class tmo_lamp_paddle: 

    def __init__(self):
        self.yag_positions = yag_positions
        self.siN_positions = siN_positions       

    def save_curr_pos(self):

        self.save_positions= {
        "magnet_x":tmo_lamp.magnet_x.position, 
        "magnet_y":tmo_lamp.magnet_y.position,  
        "magnet_z":tmo_lamp.magnet_z.position, 
        "needle_x":tmo_lamp.gas_needle_x.position, 
        "needle_y":tmo_lamp.gas_needle_y.position, 
        "needle_z":tmo_lamp.gas_needle_z.position, 
           "sq1_x":caget("MCQ:TMO:01:TARGET:X:RBV"), 
           "sq1_y":caget("MCQ:TMO:01:TARGET:Y:RBV"),  
           "sq1_z":caget("MCQ:TMO:01:TARGET:Z:RBV"),
          "sq1_rx":caget("MCQ:TMO:01:TARGET:rX:RBV"),
          "sq1_ry":caget("MCQ:TMO:01:TARGET:rY:RBV"),
          "sq1_rz":caget("MCQ:TMO:01:TARGET:rZ:RBV"),
        "laser_wp":caget("LM1K4:INJ_MP1_ATT1_WP1.RBV")}
        self.log_position("current position of chamber")
     
    def _move_to_target(self, positions):
        #block x-rays and attenuate
        im2k4.target("YAG1")
        transmission_set(0.1)
        
        #block laser
        caput("TMO:USR:BHC:AOUT:1", 0.0)
        caput("LM1K4:INJ_MP1_ATT1_WP1",positions["laser_wp"])
        
        #move needle out Y->Z->X
        tmo_lamp.gas_needle_y.mv(positions["needle_y"], wait=True)
        tmo_lamp.gas_needle_z.mv(positions["needle_z"])
        tmo_lamp.gas_needle_x.mv(positions["needle_x"])
        
        #move sq1
        caput("MCQ:TMO:01:TARGET:X", positions["sq1_x"])
        caput("MCQ:TMO:01:TARGET:Y", positions["sq1_y"])
        caput("MCQ:TMO:01:TARGET:Z", positions["sq1_z"])
        caput("MCQ:TMO:01:TARGET:rX", positions["sq1_rx"])
        caput("MCQ:TMO:01:TARGET:rY", positions["sq1_ry"])
        caput("MCQ:TMO:01:TARGET:rZ", positions["sq1_rz"])
        #Still need to add move command.....
        
        #move magnet Y->Z->X
        tmo_lamp.magnet_y.mv(positions["magnet_y"], wait=True)
        tmo_lamp.magnet_z.mv(positions["magnet_z"], wait=True)
        tmo_lamp.magnet_x.mv(positions["magnet_x"], wait=True)

    def _return_from_target(self, positions):
        
        #block x-rays
        im2k4.target("YAG1")
        
        #block laser
        caput("TMO:USR:BHC:AOUT:1", 0.0)
        
        #move magnet X->Z->Y
        tmo_lamp.magnet_x.mv(positions["magnet_x"], wait=True)
        tmo_lamp.magnet_z.mv(positions["magnet_z"], wait=True)
        tmo_lamp.magnet_y.mv(positions["magnet_y"], wait=True)
             
        #move needle out X->Z->Y
        tmo_lamp.gas_needle_x.mv(positions["needle_x"])
        tmo_lamp.gas_needle_z.mv(positions["needle_z"])
        tmo_lamp.gas_needle_y.mv(positions["needle_y"], wait=True)
        
        #move sq1
        caput("MCQ:TMO:01:TARGET:X", positions["sq1_x"])
        caput("MCQ:TMO:01:TARGET:Y", positions["sq1_y"])
        caput("MCQ:TMO:01:TARGET:Z", positions["sq1_z"])
        caput("MCQ:TMO:01:TARGET:rX", positions["sq1_rx"])
        caput("MCQ:TMO:01:TARGET:rY", positions["sq1_ry"])
        caput("MCQ:TMO:01:TARGET:rZ", positions["sq1_rz"])
        
        #unblock laser and set intensity
        caput("TMO:USR:BHC:AOUT:1", 5.0)
        caput("LM1K4:INJ_MP1_ATT1_WP1",positions["laser_wp"])
    
    def log_position(self,current_pos_string,run=None):
        curr_positions= {
        "magnet_x":tmo_lamp.magnet_x.position, 
        "magnet_y":tmo_lamp.magnet_y.position,  
        "magnet_z":tmo_lamp.magnet_z.position, 
        "needle_x":tmo_lamp.gas_needle_x.position, 
        "needle_y":tmo_lamp.gas_needle_y.position, 
        "needle_z":tmo_lamp.gas_needle_z.position, 
           "sq1_x":caget("MCQ:TMO:01:TARGET:X:RBV"), 
           "sq1_y":caget("MCQ:TMO:01:TARGET:Y:RBV"),  
           "sq1_z":caget("MCQ:TMO:01:TARGET:Z:RBV"),
          "sq1_rx":caget("MCQ:TMO:01:TARGET:rX:RBV"),
          "sq1_ry":caget("MCQ:TMO:01:TARGET:rY:RBV"),
          "sq1_rz":caget("MCQ:TMO:01:TARGET:rZ:RBV"),
        "laser_wp":caget("LM1K4:INJ_MP1_ATT1_WP1.RBV")}

        temp = {"Motors:": curr_positions.keys(), "Positions:": curr_positions.values()}
        posDF = pd.DataFrame(temp)
        msg = current_pos_string + "\n" + posDF.to_string()
        if (run is None): 
            elog.post(msg)
        else:
            elog.post(msg, run=run) 

    def target_yag(self):
        self._move_to_target(yag_positions)
        self.log_position("Nominal positions for YAG screen.")

    def target_siN(self):
        self._move_to_target(siN_positions)
        self.log_position("Nominal positions for SiN screen.")

    def target_saved(self):
        self._return_from_target()
        self.log_position("Return to stored positions.")
      
#class User:
