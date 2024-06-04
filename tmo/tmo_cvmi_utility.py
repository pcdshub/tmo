from tmo.tmo_mpods import tmo_mpod, get_mpod
from pcdsdevices.epics_motor import EpicsMotorInterface
from elog import HutchELog as elog_temp
elog = elog_temp.from_registry()
import time as time
import pandas as pd
import numpy as np
import copy
import json

#Voltages for cVMI 
#Voltages updated on 8/7/2022
#default_voltages = {
# "Plate_01":1.0,
# "Plate_02":1.0,
# "Plate_03":-4118,
# "Plate_04":-4463.8,
# "Plate_05":-2907.2,
# "Plate_06":-2141.3,
# "Plate_07":-1662.4,
# "Plate_08":-1476.1,
# "Plate_09":-1289.8,
# "Plate_10":-1103.5,
# "Plate_11":-917.2,
# "Plate_12":-730.9,
# "Plate_13":-544.6,
# "Plate_14":-451.4,
#"MCP_Front":-90,
# "MCP_Back":1810,
# "Phosphor":5010,
#}

#Positions for moving output breadboard optics
#pos_cam = {"MR1": 225, "MR2": 0, "MR3": 53.73}
#pos_mic = {"MR1": 81, "MR2": -152.5, "MR3": 200}

#3/14/23 JPC moved settings to a json file. This .json file contains a dictionary of dictionaries for the different settings. 
with open('/reg/g/pcds/pyps/apps/hutch-python/tmo/tmo/tmo_cvmi_utility.json', 'r') as f:
    settingsDict = json.load(f)

default_voltages = settingsDict['voltage_settings']
pos_cam = settingsDict['pos_cam']
pos_mic = settingsDict['pos_mic']

class CvmiVoltageUtility():
    def __init__(self):
        self.voltage_settings = default_voltages

    def shift_and_scale(self, scale=1, mcp_front=-100):
        """
        Scale and shift voltage settings for the cVMI. This function applies the scale and then shifts so that the mcp_front is at the desired voltage. This function then shows the resulting voltages. 
        Use the apply method if you are happy with the result. 
        The default voltage scale (scale=1) is set to focus 300 eV electrons at 80% of the detector radius. The scaling of energy focused at 80% and voltage is linear. 
        Parameters:
        -----------
        scale : scaling factor for voltages
        mcp_front : voltage for the mcp front. 
        """
        self._prev_voltage_settings = copy.deepcopy(self.voltage_settings)
        
        delta_Back = abs(tmo_mpod.MCP_Back.voltage.get() - tmo_mpod.MCP_Front.voltage.get())
        if delta_Back < 1:
            delta_Back = -mcp_front 
        delta_Phos = abs(tmo_mpod.Phosphor.voltage.get() - tmo_mpod.MCP_Front.voltage.get())
        if delta_Phos < 1:
            delta_Phos = -mcp_front  
        
        for k in self.voltage_settings.keys():
            self.voltage_settings[k] = scale * self.voltage_settings[k]
        
        shift = mcp_front - self.voltage_settings["MCP_Front"]
        
        for k in self.voltage_settings.keys():
            self.voltage_settings[k] = self.voltage_settings[k] + shift
    
        self.voltage_settings["Plate_01"]=self._prev_voltage_settings["Plate_01"]
        self.voltage_settings["Plate_02"]=self._prev_voltage_settings["Plate_02"]
        self.voltage_settings["MCP_Front"]=mcp_front
        #self.voltage_settings["MCP_Back"]=mcp_front + (self._prev_voltage_settings["MCP_Back"] - self._prev_voltage_settings["MCP_Front"])
        self.voltage_settings["MCP_Back"] = mcp_front + delta_Back 
        #self.voltage_settings["Phosphor"]=mcp_front + (self._prev_voltage_settings["Phosphor"] - self._prev_voltage_settings["MCP_Front"]) 
        self.voltage_settings["Phosphor"] = mcp_front + delta_Phos

        update = {
                  "Channel:": self.voltage_settings.keys(), 
          "New Voltage (V):": self.voltage_settings.values(),
          "Old Volgage (V):": self._prev_voltage_settings.values() }
        posDF = pd.DataFrame(update)
        sorted_posDF = posDF.sort_values(by=["Channel:"], ascending=True)
        print (sorted_posDF)

    def apply(self):
        """
        Apply the values stored in .voltage_settings to the corresponding mpod channels. 
        """
        for key in self.voltage_settings.keys():
            if hasattr(tmo_mpod, key):    
                mpod = getattr(tmo_mpod, key)
                mpod.set_voltage(self.voltage_settings[key])
            else:
                print('Error MPOD missing %s' %key)
            
    def replace_stored_with_current(self):
        """
        Replace the .voltage_settings with the current cVMI voltages.
        """
        self._prev_voltage_settings = copy.deepcopy(self.voltage_settings) #Retain copy in case of error
        for key in self.voltage_settings.keys():
            if hasattr(tmo_mpod, key):
                mpod = getattr(tmo_mpod, key)
                self.voltage_settings[key] = mpod.voltage_setpoint.get()
            else:
                print('Error MPOD missing %s' %key)
                self.voltage_settings = copy.deepcopy(self._prev_voltage_settings)
                return
    
    def on(self):
        """
        Turn on all named mpod channels
        """
        self.det_on(mcp_front=self.voltage_settings["MCP_Front"], mcp_back=self.voltage_settings["MCP_Back"], phosphor=self.voltage_settings["Phosphor"])
        
        for key in self.voltage_settings.keys():
            if hasattr(tmo_mpod, key):
                mpod = getattr(tmo_mpod, key)
                mpod.on()
            else:
                print('Error MPOD missing %s' %key)
            
    def off(self):
        """
        Turn off all mpod channels
        """
        mpod_key = tmo_mpod.__dict__.keys()

        for key in mpod_key:
            mpod = getattr(tmo_mpod, key)
            mpod.off()

    def det_off(self):
        """
        Turn off only detector voltages. 
        """
        tmo_mpod.Phosphor.off()
        tmo_mpod.MCP_Back.off()
        tmo_mpod.MCP_Front.off()

    def det_on(self, mcp_front=-100, mcp_back=1800, phosphor=5000):
        """
        Ramp detector to specified voltages.
        Parameters:
        -----------
        mcp_front : voltage for the mcp front.
        mcp_back : voltage for the mcp back.
        phosphor : voltage for the phosphor.  
        """
        delta_back = mcp_back - mcp_front
        delta_phos = phosphor - mcp_front
        
        tmo_mpod.MCP_Front.set_voltage(mcp_front)
        tmo_mpod.MCP_Front.on()
        tmo_mpod.MCP_Back.set_voltage(mcp_front + 0.2*delta_back)
        tmo_mpod.Phosphor.set_voltage(mcp_front + 0.2*delta_phos)
        tmo_mpod.MCP_Back.on()
        tmo_mpod.Phosphor.on()        

        time.sleep(5)
        if (tmo_mpod.MCP_Front.state.get() == 'on'):
            self.voltage_settings["MCP_Front"] = mcp_front
        else:
            print('Failed to ramp MCP Front')
            return 0 
        
        if (tmo_mpod.MCP_Back.state.get() == 'on'):
            self.voltage_settings["MCP_Back"] = mcp_back
        else:
            print('Failed to ramp MCP Back')
            return 0
        
        if (tmo_mpod.Phosphor.state.get() == 'on'):
            while ( tmo_mpod.MCP_Back.voltage.get() < (0.9 * tmo_mpod.MCP_Back.voltage_setpoint.get() ) ):
                time.sleep(2)
            while ( tmo_mpod.Phosphor.voltage.get() < (0.9 * tmo_mpod.Phosphor.voltage_setpoint.get() ) ):
                time.sleep(2)

            tmo_mpod.MCP_Back.set_voltage(mcp_back)
            tmo_mpod.Phosphor.set_voltage(phosphor)
            self.voltage_settings["Phosphor"] = phosphor    
        else:
            print('Failed to ramp Phosphor')
            return 0

    def checkout(voltage=100):
        """
        Set all cVMI channels to the voltage specified and turn them on.
        Parameters:
        -----------
        voltage : voltage to set all channels for testing.  
        """
        for key in self.voltage_settings.keys():
            if hasattr(tmo_mpod, key):
                mpod = getattr(tmo_mpod, key)
                mpod.set_voltage(voltage)
                mpod.on()
            else:
                print('Error MPOD missing %s' %key)

        get_mpod()
        
cvmi_voltage_utility = CvmiVoltageUtility()

class CvmiOptics:

    def __init__(self):
        #with open('/reg/g/pcds/pyps/apps/hutch-python/tmo/tmo/tmo_cvmi_utility_positions.json', 'r') as handle:
        #    self.voltage_settings = json.loads(json_file)
        self.cam_positions = pos_cam
        self.mic_positions = pos_mic
        self.cvmi_out_mr1 = EpicsMotorInterface('LM1K4:EJX_DP6_CVS1_MR1', name='cvmi_out_mr1')
        self.cvmi_out_mr2 = EpicsMotorInterface('LM1K4:EJX_DP6_CVS1_MR2', name='cvmi_out_mr2')
        self.cvmi_out_mr3 = EpicsMotorInterface('LM1K4:EJX_DP6_CVS1_MR3', name='cvmi_out_mr3')

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
        self.cvmi_out_mr1.umv(pos_cam["MR1"])
        time.sleep(2)
        if (self.cvmi_out_mr1.wm() < (pos_cam["MR1"] - 1) ): 
            return None        

        #move MR2
        self.cvmi_out_mr2.umv(pos_cam["MR2"])
        time.sleep(2)
        if (self.cvmi_out_mr2.wm() < (pos_cam["MR2"] - 1)):
            return None
        
        #move MR3
        self.cvmi_out_mr3.umv(pos_cam["MR3"])
        time.sleep(2)
        if (self.cvmi_out_mr3.wm() > (pos_cam["MR3"] + 1) ):
            return None
        
        
    def move_to_mic(self):     
        #move MR3
        self.cvmi_out_mr3.umv(pos_mic["MR3"])
        time.sleep(2)
        if (self.cvmi_out_mr3.wm() < (pos_mic["MR3"] - 1)):
            return None
        
        #move MR2
        self.cvmi_out_mr2.umv(pos_mic["MR2"])
        time.sleep(2)
        if (self.cvmi_out_mr2.wm() > (pos_mic["MR2"] + 1)):
            return None 

        #move MR1
        self.cvmi_out_mr1.umv(pos_mic["MR1"])
        time.sleep(2)
        if (self.cvmi_out_mr1.wm() > (pos_mic["MR1"] + 1)):
            return None

cvmi_optics = CvmiOptics()         
