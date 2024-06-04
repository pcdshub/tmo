from hutch_python.utils import safe_load
from tmo.tmo_laser import LaserMirrorMotor
from pcdsdevices.epics_motor import EpicsMotorInterface
from tmo.db import tm1k4
from tmo.tmo_photon_energy import hf_w
from tmo.tmo_mpods import tmo_mpod
from elog import HutchELog as elog_temp
from tmo.db import tmo_cvmi
elog = elog_temp.from_registry()

import pandas as pd
import numpy as np
import time as time

pos_cam = {"MR1": 225, "MR2": 0, "MR3": 53.73}
pos_mic = {"MR1": 81, "MR2": -152.5, "MR3": 200}

pos_cell_yag = {"gas_cell_x": -15.05, "gas_cell_y": 78.0, "gas_cell_z": 9.389}
pos_cell_front = {"gas_cell_x": -15.05, "gas_cell_y": 67.6, "gas_cell_z": 12.5}
pos_cell_center = {"gas_cell_x": -15.05, "gas_cell_y": 67.6, "gas_cell_z": 9.389}
pos_cell_out = {"gas_cell_x": -15.05, "gas_cell_y": 85.0, "gas_cell_z": 9.389}

pos_cell = {"yag": pos_cell_yag, "front": pos_cell_front, "center":pos_cell_center, "out":pos_cell_out}  


class User:

    def __init__(self):
        self.cam_positions = pos_cam
        self.mic_positions = pos_mic
        self.cvmi_out_mr1 = EpicsMotorInterface('LM1K4:EJX_DP6_CVS1_MR1', name='cvmi_out_mr1')
        self.cvmi_out_mr2 = EpicsMotorInterface('LM1K4:EJX_DP6_CVS1_MR2', name='cvmi_out_mr2')
        self.cvmi_out_mr3 = EpicsMotorInterface('LM1K4:EJX_DP6_CVS1_MR3', name='cvmi_out_mr3')
        self.gas_cell_positions = pos_cell

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

    def move_gas_cell(self, pos):
        """
        Method to move to the gas cell between different positions. The positions are saved in the dictionary .gas_cell_positions. 
        Parameters:
        --------------
        pos: string indicating the position you hope to move to, for example "yag", or "out". If an invalid position is specified the list of possible positions is retruned.
        """
        if pos in self.gas_cell_positions.keys():
            tmo_cvmi.gas_cell_x.umv(self.gas_cell_positions[pos]["gas_cell_x"])
            tmo_cvmi.gas_cell_z.umv(self.gas_cell_positions[pos]["gas_cell_z"])
            tmo_cvmi.gas_cell_y.umv(self.gas_cell_positions[pos]["gas_cell_y"])
        else:
            print("position not in keys, possible positions:")
            for key in self.gas_cell_positions.keys():
                print(key)

    def get_gas_cell(self, log=False, msg=None):
        """
        Method to get the current position of the gas cell. This can also be used to log the position in the log book.
        Parameters:
        --------------
        log: optional boolean. log=True will post the positions to the elog
        msg: optional string. When logging to the elog, the string in msg will be attached to the post. 
        """
        curr_positions= {
            "gas_cell_x":tmo_cvmi.gas_cell_x.position,
            "gas_cell_y":tmo_cvmi.gas_cell_y.position,
            "gas_cell_z":tmo_cvmi.gas_cell_z.position, }

        temp = {"Motors:": curr_positions.keys(), "Positions:": curr_positions.values()}
        posDF = pd.DataFrame(temp)
        print(posDF)
        
        if (log is True):
            if (msg is not None):
                msg = msg + "\n" + posDF.to_string()
            else:
                msg = posDF.to_string()
            elog.post(msg)

    def checkATM(self, photonEnergy=850):
        """
        Move the ATM paddle into the beam and adjust the photon energy below the absorption. 
        Reduce the MCP voltages 
        """
        elog.post('moving ATM and photon energy to check timing on ATM.')
        hf_w.mv(photonEnergy)
        tm1k4.y_motor.mv(-16.318)
        tmo_mpod.Phosphor.set_voltage(4700)
        tmo_mpod.MCP_Back.set_voltage(1700)
        

    def goToDataTaking(self, photonEnergy):
        """
        Move the ATM paddle to the edge of the beam and move the photon energy.
        """
        elog.post('Returning to the data taking positions for the ATM paddle. Photon Energy at ' + str(hf_w.position)) 
        tm1k4.y_motor.mv(-17.345)
        hf_w.mv(photonEnergy)
        if (photonEnergy > 860):
            tmo_mpod.MCP_Back.set_voltage(1700)
            tmo_mpod.Phosphor.set_voltage(4700)
        else:
            tmo_mpod.MCP_Back.set_voltage(1700)
            tmo_mpod.Phosphor.set_voltage(4700)
               
           



