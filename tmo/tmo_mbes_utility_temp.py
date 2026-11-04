from tmo.tmo_mpods import tmo_mpod, get_mpod
from pcdsdevices.epics_motor import EpicsMotorInterface
from ophyd.signal import EpicsSignal, EpicsSignalRO
from pcdsdevices.interface import BaseInterface
from ophyd.device import Component as Cpt, Device
from pcdsdevices.valve import VCN_VAT590
from tmo.tmo_beamline_control import _log_positions
#from elog import HutchELog as elog_temp
#elog = elog_temp.from_registry()
import time as time
import pandas as pd
import numpy as np
import copy
import json
from epics import caget, caput, cainfo


# default_voltages = {
# #"Coil_1_Ret":0,
# #"Coil_2_Ret":0,
# "coil_bias":-50,
# "mbes_ft_lens":0,
# "mbes_ip_lens":0,
# "ion_tof_flight_tube":-4650,
# "ion_tof_mcp_b":-3800,
# "ion_tof_mcp_f":-4850,
# "ion_tof_pmt":-700,
# "ion_tof_scint":-1500,
# "mbes_mcp_back":100,
# "mbes_mcp_f":100,
# "mbes_mcp_grid":0,
# "magnet":-178,
# #"Needle":0
# }

with open('/reg/g/pcds/pyps/apps/hutch-python/tmo/tmo/tmo_mpod_configs/voltages_mbes.json', 'r') as f:
    default_voltages = json.load(f)

class LampVCN(BaseInterface, Device):
    """
    LAMP remote controled leak valve. 
    """
    tab_component_names = True
    tab_whitelist = ['open_valve', 'close_valve','set_manual_cntl','set_pres_cntl','status']


    valve = Cpt(VCN_VAT590, 'LAMP:VCN:01:' )
    
    def open_valve(self):
        """
        Open the valve to the opening limit
        """
        self.valve.state.put(2)
        
    def close_valve(self):
        """
        Close the valve
        """
        self.valve.state.put(1)
        
    def set_manual_cntl(self, set_point=0):
        """
        Set the valve to manual control and open to set_point

        Parameters
        ----------
        set_point : float, optional
            Percentage to open the valve. The default is 0.

        """
        self.valve.pos_req.put(0)
        self.valve.state.put(0)
        self.valve.pos_req.put(set_point)
        
    def set_pres_cntl(self, set_point=0):
        """
        Set the valve to pressure control and set the control pressure to set_point

        Parameters
        ----------
        set_point : float, optional
            Pressure setpoint in Torr. The default is 0.

        """
        self.valve.pres_req.put(0)
        self.valve.state.put(3)
        self.valve.pres_req.put(set_point)
        
    def status(self, log=False, msg=None, **kwargs):
        """
        Print the status of the valve

        Parameters
        ----------
        log : boll, optional
            Flag to post the status to elog. The default is False.
        msg : str, optional
            Message to include in elog post. The default is None.
        run : int, optional
            Associate the post with a specific run
        tags : list, optional
            List of tags to add to the post

        """
        valve_states = ['ManualControl', 'CloseValve', 'OpenValve', 'PressureControl', 'HoldValve']
        status_dict={
            'control mode': valve_states[self.valve.state.get()],
            'pressure setting (10-8 torr)': self.valve.pres_req.get()*1E8,
            'pressure reading (10-8 torr)': self.valve.pres_torr.get()*1E8,
            'position setting': self.valve.pos_raw.get()
            }
        _log_positions(status_dict, log, msg, **kwargs)
       
tmo_lamp_vcn = LampVCN('',name='tmo_lamp_vcn')

def get_lamp_vcn(log=False, msg=None, **kwargs):
        """
        Print the status of the remote controlled VAT-590 valve

        Parameters
        ----------
        log : boll, optional
            Flag to post the status to elog. The default is False.
        msg : str, optional
            Message to include in elog post. The default is None.
        run : int, optional
            Associate the post with a specific run
        tags : list, optional
            List of tags to add to the post

        """
        tmo_lamp_vcn.status(log, msg, **kwargs)

class PDU(BaseInterface, Device):
    tab_component_names= True
    tab_whitelist = ['on', 'off']

    _on_switch = Cpt(EpicsSignal, ':setPduOutletCommand', kind='omitted')
 #   _off_switch = Cpt(EpicsSignal, ':setPduOutletCommand', kind='omitted')
    status = Cpt(EpicsSignalRO, ':getPduOutletState', kind = 'hinted', string=True)
    ch_name = Cpt(EpicsSignal, ':setPduOutletName', kind='normal', string=True)
    crtl_status = Cpt(EpicsSignalRO, ':getPduOutletControllable', kind='omitted')

    def on(self):
        self._on_switch.put(2)
    def off(self):
        self._on_switch.put(1)

ortec_ps = PDU('TMO:PWR:CVMI:Outlet:3', name='ortec_ps')


class mbes_Voltage_utility(): 
    
    """
    This class provides utility functions to manage and control voltage settings for ion and electron components in 
    an experimental setup. It categorizes the components into ion and electron keys and provides methods to 
    turn them on/off, ramp voltages, and set specific voltage values.
    
    Some example workflow:
        * mbes_voltage.ramp_electron(ret=100) to ramp the electron side only with 100 V retardation. Note
        the voltage difference across the MCP front and back is 1000 V if this is invoked for the first time.
        For subsequent occurances, this ramp_electron keeps track of the voltage difference between the two MCPs and
        set that following any retardation value change.
        
        * mbes_voltage.ramp_ion() to ramp the ion side only according to a predefined settings.
        
        * mbes_voltage.ramp_ion_electron(ret=100) to do a coordinated ramping of voltage for both the 
        ion and the electron side with 100 V retardation. This will adjust the nominal voltage for the grid and FT lens to
        make sure that the electrons which are accelerated due to magnet voltage are retarded accordingly.
        
        * mbes_voltage.electron_mcp_diff(1000)
        This will make sure that the voltage between the back and the front MCP for the MBES electron side has the
        specified differences.
        
    
    """
    
    
    def __init__(self):
        
        """
        Initializes the voltage utility with default settings. It classifies the voltage keys into ion and electron 
        components based on their names.
        """
        
        self.voltage_settings = default_voltages
        mpod_keys = self.voltage_settings
        ion_keys = []
        electron_keys = []
        
        # Classify keys into ion and electron components
        for keys in mpod_keys:
            if 'ion'.casefold() in keys.casefold():
                ion_keys.append(keys)
            else: electron_keys.append(keys)
        self.ion_keys = ion_keys
        self.electron_keys = electron_keys

    def on(self):

        """
        Turns on all components listed in the voltage settings
        """
        tmo_mpod.clear_faults_all()
        mpod_keys = self.voltage_settings
        for keys in mpod_keys:
            mpodOn = getattr(tmo_mpod, keys)
            print('Turning on %s'%keys)
            mpodOn.on()
            
    def off(self):
        """
        Turns off all components listed in the voltage settings.
        """
        self.amplifier('OFF')
        mpod_keys = self.voltage_settings
        for keys in mpod_keys:
            mpodOn = getattr(tmo_mpod, keys)
            print('Turning off %s'%keys)
            mpodOn.off()
        tmo_mpod.clear_faults_all()

    def amplifier(self, state='ON'):
        
        """
        Turns the amplifier on or off based on the specified state.
        
        Args:
            state (str): 'ON' to turn the amplifier on, 'OFF' to turn it off.
        Usage: amplifier('ON') to turn off the ortec amplifier
        """
        
        if state=='ON':
            ortec_ps.on() #caput('TMO:PWR:01:Outlet:6:TurnOn', 1)
            caput('TMO:USR:BHC:BOUT:1', 1)
            time.sleep(3)
        elif state=='OFF':
            ortec_ps.off()#caput('TMO:PWR:01:Outlet:6:TurnOff', 1)
            caput('TMO:USR:BHC:BOUT:1', 0)
            time.sleep(3)
            
    def set_to_zero_ion(self):
        
        """
        Sets the voltage of all ion components to zero and turns off the amplifier.
        """
        
        self.amplifier('OFF')
        
        
        for k in self.ion_keys:
            if 'flight_tube'.casefold() in k.casefold():
                self.ion_FT = getattr(tmo_mpod, k)
                ion_FTVoltage = self.voltage_settings[k]
                
            elif 'mcp_b'.casefold() in k.casefold():
                self.ion_mcpBack = getattr(tmo_mpod, k)
                ion_mcpBackVoltage = self.voltage_settings[k]
                
            elif 'mcp_f'.casefold() in k.casefold():
                self.ion_mcpFront = getattr(tmo_mpod, k)
                ion_mcpFrontVoltage = self.voltage_settings[k]
                
            elif 'pmt'.casefold() in k.casefold():
                self.ion_pmt= getattr(tmo_mpod, k)
                pmtVoltage = self.voltage_settings[k]
                
            elif 'scint'.casefold() in k.casefold():
                self.ion_scn= getattr(tmo_mpod, k)
                scintillatorVoltage = self.voltage_settings[k]
            
        
        self.ion_pmt.voltage.put(0)
        if abs(scintillatorVoltage) > abs(pmtVoltage):
            self.ion_scn.voltage.put(scintillatorVoltage - pmtVoltage)
            self.ion_mcpFront.voltage.put(ion_mcpFrontVoltage - pmtVoltage)
            self.ion_mcpBack.voltage.put(ion_mcpBackVoltage - pmtVoltage)
            self.ion_FT.voltage.put(ion_FTVoltage - pmtVoltage)
        
        time.sleep(4)
        

        self.ion_scn.voltage.put(0)
        
        if abs(ion_mcpFrontVoltage ) > abs(scintillatorVoltage):
            self.ion_mcpFront.voltage.put(ion_mcpFrontVoltage - scintillatorVoltage)
            self.ion_mcpBack.voltage.put(ion_mcpBackVoltage - scintillatorVoltage)
            self.ion_FT.voltage.put(ion_FTVoltage - scintillatorVoltage)

        time.sleep(4)
        
        
        self.ion_mcpFront.voltage.put(0)
        
        if abs(ion_mcpBackVoltage ) > abs(ion_mcpFrontVoltage):
            self.ion_mcpBack.voltage.put(ion_mcpBackVoltage - ion_mcpFrontVoltage)
            self.ion_FT.voltage.put(ion_FTVoltage - ion_mcpFrontVoltage)

        time.sleep(8)
        

        
        for keys in self.ion_keys:
            voltObj = getattr(tmo_mpod, keys)
            self.ramp_volt(voltObj, 0)
        voltObj = getattr(tmo_mpod, 'magnet')
        self.ramp_volt(voltObj, 0)



    def set_to_zero_electron(self):
        
        """
        Sets the voltage of all electron components to zero and turns off the amplifier.
        """
        
        self.amplifier('OFF')
        for keys in self.electron_keys:
            voltObj = getattr(tmo_mpod, keys)
            self.ramp_volt(voltObj, 0)
        tmo_mpod.clear_faults_all()
        
    def wait_self(self, volt_object, voltageToCheck):
        """
        Waits until the voltage of the specified object reaches the target value within a tolerance.
        
        Args:
            volt_object: The voltage object to monitor.
            voltageToCheck (float): The target voltage to reach.
        """
        
        
        while (np.abs(volt_object.voltage.get()) < np.abs(voltageToCheck) - 0.05 * np.abs(voltageToCheck)):
            time.sleep(0.1)
    
    def ramp_volt(self, volt_object, voltage, wait=1):
        
        """
        Ramps the voltage of the specified object to the target value and optionally waits for the voltage to stabilize.
        
        Args:
            volt_object: The voltage object to modify.
            voltage (float): The target voltage.
            wait (int): Whether to wait for the voltage to stabilize (1) or not (0).
        """
        
        volt_object.voltage.put(voltage)
        if wait==1:
            self.wait_self(volt_object, voltage)
            
    def get_volt(self, volt_object):
        """
        Retrieves the current voltage of the specified object.
        
        Args:
            volt_object: The voltage object to query.
        
        Returns:
            float: The current voltage of the object.
        """
                
        return volt_object.voltage.get()
            
    def voltageStepSize(self, initial, final, stepSize=100.0):
        
        """
        Calculates the steps needed to ramp the voltage from the initial to the final value.
        
        Args:
            initial (float): The initial voltage.
            final (float): The final voltage.
            stepSize (float): The size of each voltage step.
        
        Returns:
            ndarray: An array of voltage steps.
        """
                
        
        vstep = int(np.round((final - initial)/stepSize))
        volt_steps = np.linspace(initial, final, np.abs(vstep))
        return volt_steps
    
    def ramp_ion(self):
        
        """
        Ramps the voltage of all ion components according to the predefined settings.
        """
        
        self.on()
        self.amplifier('OFF')
        time.sleep(1)
        for k in self.ion_keys:
            if 'flight_tube'.casefold() in k.casefold():
                self.ion_FT = getattr(tmo_mpod, k)
                ion_FTVoltage = self.voltage_settings[k]
                
            elif 'mcp_b'.casefold() in k.casefold():
                self.ion_mcpBack = getattr(tmo_mpod, k)
                ion_mcpBackVoltage = self.voltage_settings[k]
                
            elif 'mcp_f'.casefold() in k.casefold():
                self.ion_mcpFront = getattr(tmo_mpod, k)
                ion_mcpFrontVoltage = self.voltage_settings[k]
                
            elif 'pmt'.casefold() in k.casefold():
                self.ion_pmt= getattr(tmo_mpod, k)
                pmtVoltage = self.voltage_settings[k]
                
            elif 'scint'.casefold() in k.casefold():
                self.ion_scn= getattr(tmo_mpod, k)
                scintillatorVoltage = self.voltage_settings[k]
            
        
        self.ion_pmt.voltage.put(pmtVoltage)
        #self.ion_scn.voltage.put(pmtVoltage)  #The PS only can do +ve 06/21/26
        self.ion_scn.voltage.put(0)
        self.ion_mcpFront.voltage.put(pmtVoltage)
        self.ion_mcpBack.voltage.put(pmtVoltage)
        self.ion_FT.voltage.put(pmtVoltage)
        
        self.wait_self(self.ion_FT, pmtVoltage)
        """        
        #ramping everything scintillator onwards to Ion Flightube############

        volt_steps = self.voltageStepSize(pmtVoltage, scintillatorVoltage, 50)
        
        
        for volts in volt_steps:
            self.ramp_volt(self.ion_scn, volts, 0)
            self.ramp_volt(self.ion_mcpFront, volts, 0)
            self.ramp_volt(self.ion_mcpBack, volts, 0)
            self.ramp_volt(self.ion_FT, volts, 1)
        """
        self.ramp_volt(self.ion_scn, scintillatorVoltage)
        ######################################################################    
        

        #volt_steps = self.voltageStepSize(scintillatorVoltage, ion_mcpBackVoltage, 50)
        volt_steps = self.voltageStepSize(pmtVoltage, ion_mcpBackVoltage, 50)
        print('Step 1')
        #print(volt_steps)
        for volts in volt_steps:
            self.ramp_volt(self.ion_mcpBack, volts, 0)
            self.ramp_volt(self.ion_mcpFront,volts, 0)
            self.ramp_volt(self.ion_FT, volts, 1)
        self.ramp_volt(self.ion_mcpBack, ion_mcpBackVoltage)
        #######################################################################
        if np.abs(ion_mcpFrontVoltage) > np.abs(ion_FTVoltage):
            volt_steps = self.voltageStepSize(ion_mcpBackVoltage, ion_FTVoltage, 50)
            #print('Step 2 if')
            #print(volt_steps)
            for volts in volt_steps:
                self.ramp_volt(self.ion_mcpFront,volts, 0)
                self.ramp_volt(self.ion_FT, volts, 1)
            self.ramp_volt(self.ion_FT, ion_FTVoltage)
            self.ramp_volt(self.ion_mcpFront, ion_mcpFrontVoltage, 1)
        else: 
            volt_steps = self.voltageStepSize(ion_mcpBackVoltage, ion_mcpFrontVoltage, 50)
            #print('Step 2 else')
            #print(volt_steps)
            for volts in volt_steps:
                self.ramp_volt(self.ion_mcpFront,volts, 0)
                self.ramp_volt(self.ion_FT, volts, 1)
            self.ramp_volt(self.ion_mcpFront, ion_mcpFrontVoltage)
            self.ramp_volt(self.ion_FT, ion_FTVoltage, 1)
        
        for k in self.electron_keys:
            if 'magnet'.casefold() in k.casefold():
                self.ion_magnet = getattr(tmo_mpod, k)
                ion_magnetVoltage = self.voltage_settings[k]
        self.ramp_volt(self.ion_magnet, ion_magnetVoltage, 1)
        
        self.amplifier('ON')
        time.sleep(1)
        print('Ion Ramp Done!')
        

    def ramp_electron(self, ret=0):
        """
        Ramps the voltage of all electron components according to the predefined settings and specified retardation.
        
        Args:
            ret (float): The retardation voltage to apply for the electron.
        Example:
        mbes_voltage.ramp_electron(ret=100) for 100 V retardation. 
        It doesn't matter the sign of the ret value. The ret voltage will ALWYAS be negative.
        
        """
        

        self.on()

        ret = -1 * np.abs(ret)
        coil_bias = np.max([0.5*ret, -200])
        FT_lensVoltage = ret
        mcp_gridVoltage = ret
        
        for k in self.electron_keys:
            if 'mcp_f'.casefold() in k.casefold():
                self.electron_mcpFront = getattr(tmo_mpod, k)
                electron_mcpFrontVoltage = self.voltage_settings[k]
                
            elif 'back'.casefold() in k.casefold():
                self.electron_mcpBack = getattr(tmo_mpod, k)
                electron_mcpBackVoltage = self.voltage_settings[k]
                
            elif 'grid'.casefold() in k.casefold():
                self.electron_mcpGrid = getattr(tmo_mpod, k)
                electron_mcpGridVoltage = mcp_gridVoltage
                
            elif 'ft_lens'.casefold() in k.casefold():
                self.electron_FT= getattr(tmo_mpod, k)
                electron_FTVoltage = FT_lensVoltage
                
            elif 'ip'.casefold() in k.casefold():
                self.electron_IP= getattr(tmo_mpod, k)
                electron_IPVoltage = self.voltage_settings[k]
                
# =============================================================================
#             elif 'Coil_1'.casefold() in k.casefold():
#                 self.electron_coil_1 = getattr(tmo_mpod, k)
#                 electron_coil_1_Voltage = coil_bias
#             
#             elif 'Coil_2'.casefold() in k.casefold():
#                 self.electron_coil_2 = getattr(tmo_mpod, k)
#                 electron_coil_2_Voltage = coil_bias
# =============================================================================
            elif 'coil_bias'.casefold() in k.casefold():
                self.electron_coil_1 = getattr(tmo_mpod, k)
                electron_coil_1_Voltage = coil_bias

            elif 'needle'.casefold() in k.casefold():
                self.needle = getattr(tmo_mpod, k)
                electron_needleVoltage = self.voltage_settings[k]
        
        
        mcpBack_Read = np.round(self.get_volt(self.electron_mcpBack))
        mcpFront_Read = np.round(self.get_volt(self.electron_mcpFront))
        mcpDiff = mcpBack_Read - mcpFront_Read
        print(mcpDiff)
        
        if mcpDiff < 1:
            mcpDiff = 1000
            ret_Read = electron_FTVoltage
        else: ret_Read = self.get_volt(self.electron_FT)
        
        mcpBackTemp = mcpFront_Read + 1000
        self.ramp_volt(self.electron_mcpBack, mcpBackTemp, 1) # step 1
        
        self.amplifier(state='OFF')
        time.sleep(2)                  # turn off amlifier Step 2 and 3
        #step 4, 5, 6
        print('Starting ramping mcps, coil, and grid...')
        if np.abs(ret) < np.abs(ret_Read):
            self.ramp_volt(self.electron_FT, electron_FTVoltage, 0)
            self.ramp_volt(self.electron_mcpGrid, electron_mcpGridVoltage, 0)
            self.ramp_volt(self.electron_coil_1, coil_bias, 0)
            # self.ramp_volt(self.electron_coil_2, coil_bias, 0)
            
            self.ramp_volt(self.electron_mcpFront, ret + 300)
            self.ramp_volt(self.electron_mcpBack, ret + 300 + 1000)
        else:
            self.ramp_volt(self.electron_mcpFront, ret + 300)
            self.ramp_volt(self.electron_mcpBack, ret + 300 + 1000)
            self.ramp_volt(self.electron_coil_1, coil_bias, 0)
            # self.ramp_volt(self.electron_coil_2, coil_bias, 0)
            self.ramp_volt(self.electron_mcpGrid, electron_mcpGridVoltage, 0)
            self.ramp_volt(self.electron_FT, electron_FTVoltage, 0)
            
        self.amplifier(state='ON')
        print('Doing the final stage of ramping...')
        time.sleep(2) 
        self.ramp_volt(self.electron_mcpBack, ret + 300 + mcpDiff)
        print('electron side ramping Done. Finished all steps')


    def ramp_ion_electron(self, ret=0):
        
        """
        Ramp the voltage for both ion and electron components according to the predefined settings.
        
        Parameters:
        ret (float): The retardation voltage for the electron side. Default is 0.
        
        Example:
        mbes_voltage.ramp_ion_electron(100),
        This will apply a 100 V retardation on top of the acceleration that the electrons are experiencing due to Magnet voltage.
        
        """
                
        
        self.ramp_ion()
        
        voltObjMaG = getattr(tmo_mpod, 'magnet')
        for k in self.electron_keys:
            if 'ip'.casefold() in k.casefold():
                voltObjIP = getattr(tmo_mpod, k)
        magnetVal = voltObjMaG.voltage.setpoint
        IPVal = voltObjIP.voltage.setpoint
        addedRet = np.abs((IPVal - magnetVal)/2)
        
        self.ramp_electron(np.abs(ret) + addedRet)
        print('Ramping complete for both ion and electron side. Make sure the mcp voltage difference of the electron side is what you wanted. If not, use mbes_voltage.electron_mcp_diff() to change it.')
        
    def electron_mcp_diff(self, mcpDiff=1000):
        
        for k in self.electron_keys:
            if 'f'.casefold() in k.casefold():
                self.electron_mcpFront = getattr(tmo_mpod, k)
                electron_mcpFrontVoltage = self.voltage_settings[k]
                
            elif 'back'.casefold() in k.casefold():
                self.electron_mcpBack = getattr(tmo_mpod, k)
                electron_mcpBackVoltage = self.voltage_settings[k]
                
        mcpBack_Read = np.round(self.get_volt(self.electron_mcpBack))
        mcpFront_Read = np.round(self.get_volt(self.electron_mcpFront))
        mcpDiff_now = mcpBack_Read - mcpFront_Read
        
        if np.abs(mcpDiff) > 2000:
            print('Too high of a voltage difference. Make sure that the voltage difference is below 2 kV.')
        elif np.abs(mcpDiff) != np.abs(mcpDiff_now):
            self.ramp_volt(self.electron_mcpBack, mcpFront_Read + np.abs(mcpDiff))
            print('Done ramping MCP Back of the electron side to get the desired diff.')
        else:
            print('Nothing to do here! The difference is at 1 kV which is a volt diff between back and the front.')

mbes_voltage = mbes_Voltage_utility()
