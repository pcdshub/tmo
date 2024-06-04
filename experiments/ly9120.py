from hutch_python.utils import safe_load
import numpy as np
import time as time
from elog import HutchELog as elog_temp
elog = elog_temp.from_registry()
import pandas as pd

pos_cam = {"MR1": 299, "MR2": -27.39, "MR3": -92.8}
pos_mic = {"MR1": 226.33, "MR2": -130.39, "MR3": 0}

with safe_load('cVMI Outcoupling Motors'):
    from pcdsdevices.pv_positioner import PVPositionerIsClose
    from ophyd.device import Component as Cpt
    from ophyd.signal import EpicsSignal, EpicsSignalRO
    class LaserMirrorMotor(PVPositionerIsClose):
        setpoint = Cpt(EpicsSignal, '')
        readback = Cpt(EpicsSignalRO, '.RBV')
        atol = 0.1
        rtol = 0
    
class TmoLaser:
    def __init__(self):
        self.lens_x = LaserMirrorMotor('LM1K4:EJX_MP2_LS3_LM2', name='lens_x')
        self.lens_y = LaserMirrorMotor('LM1K4:EJX_MP2_LS3_LM3', name='lens_y')
        self.lens_z = LaserMirrorMotor('LM1K4:EJX_MP2_LS3_LM1', name='lens_z')
        self.inj_mr = LaserMirrorMotor('LM1K4:INJ_DP2_MR1', name='inj_mr')

tmo_laser = TmoLaser()

class User:

    def __init__(self):
        self.cam_positions = pos_cam
        self.mic_positions = pos_mic
        self.cvmi_out_mr1 = LaserMirrorMotor('LM1K4:EJX_DP6_CVS1_MR1', name='cvmi_out_mr1')
        self.cvmi_out_mr2 = LaserMirrorMotor('LM1K4:EJX_DP6_CVS1_MR2', name='cvmi_out_mr2')
        self.cvmi_out_mr3 = LaserMirrorMotor('LM1K4:EJX_DP6_CVS1_MR3', name='cvmi_out_mr3')

    def get_cvmi_mr(self):
        curr_positions= {
        "MR1": self.cvmi_out_mr1.position, 
        "MR2": self.cvmi_out_mr2.position,  
        "MR3": self.cvmi_out_mr3.position}

        temp = {"Motors:": curr_positions.keys(), "Positions:": curr_positions.values()}
        posDF = pd.DataFrame(temp)
        print(posDF)

    def move_to_cam(self):
        #move MR1
        self.cvmi_out_mr1.mv(pos_cam["MR1"], wait=True)
        time.sleep(2)
        if (self.cvmi_out_mr1.wm() < (pos_cam["MR1"] - 1) ): 
            return None        

        #move MR2
        self.cvmi_out_mr2.mv(pos_cam["MR2"], wait=True)
        time.sleep(2)
        if (self.cvmi_out_mr2.wm() < (pos_cam["MR2"] - 1)):
            return None
        
        #move MR3
        self.cvmi_out_mr3.mv(pos_cam["MR3"], wait=True)
        time.sleep(2)
        if (self.cvmi_out_mr3.wm() > (pos_cam["MR3"] + 1) ):
            return None
        
        
    def move_to_mic(self):     
        #move MR3
        self.cvmi_out_mr3.mv(pos_mic["MR3"], wait=True)
        time.sleep(2)
        if (self.cvmi_out_mr3.wm() < (pos_mic["MR3"] - 1)):
            return None
        
        #move MR2
        self.cvmi_out_mr2.mv(pos_mic["MR2"], wait=True)
        time.sleep(2)
        if (self.cvmi_out_mr2.wm() > (pos_mic["MR2"] + 1)):
            return None 

        #move MR1
        self.cvmi_out_mr1.mv(pos_mic["MR1"], wait=True)
        time.sleep(2)
        if (self.cvmi_out_mr1.wm() > (pos_mic["MR1"] + 1)):
            return None



