####### 
# JPC was playing with a better way to build the vacuum parts
# Not sure it is worth it
########
from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from pcdsdevices.gauge import GCCPLC, GaugePLC
from pcdsdevices.pump import PTMPLC, PIPPLC
from pcdsdevices.valve import VGC_2S, VCN
from ophyd.device import Component as Cpt
from tmo.db import tv1k4_vgc_01, tv1k4_vfs_01, tv1k4_vgc_02, st3k4_pps_vgc_01, mr2k4_kbo_vwc_01, mr3k4_kbo_vwc_01, mr3k4_kbo_vfs_01, li3k4_ip1_vwc_01, pc5k4_vwc_01, mr4k4_kbo_vwc_01, mr5k4_kbo_vwc_01, tm2k4_kbo_vwc_01
import pandas as pd
from collections import OrderedDict
import logging
import json
import re

logger = logging.getLogger(__name__)

with open('/reg/g/pcds/pyps/apps/hutch-python/tmo/tmo/tmo_vac_utility.json', 'r') as f:
    vacDict = json.load(f)

# HV Gauges on the beamline
_HV_BL_GAUGES = vacDict['HV_BL_GAUGES']

#LV Gagues on the beamline
_LV_BL_GAUGES = vacDict['LV_BL_GAUGES']

#HV LAMP Gauges
_HV_LAMP_GAUGES = vacDict['HV_LAMP_GAUGES']

#LV LAMP Gauges
_LV_LAMP_GAUGES = vacDict['LV_LAMP_GAUGES']

#HV MRCO Gauges
_HV_MRCO_GAUGES = vacDict['HV_MRCO_GAUGES']

#LV MRCO Gauges
_LV_MRCO_GAUGES = vacDict['LV_MRCO_GAUGES']

#HV DREAM Gauges
_HV_DREAM_GAUGES = ['DREAM:NC:GCC:01','DREAM:DP1:GHC:01','DREAM:DP2:GHC:01','DREAM:DP3:GHC:01']

#LV DREAM Gauges
_LV_DREAM_GAUGES = ['DREAM:NC:GPI:01','DREAM:DP1:GPI:01']

# All Gauges
_GAUGES = [_HV_BL_GAUGES, _LV_BL_GAUGES, _HV_LAMP_GAUGES, _LV_LAMP_GAUGES, _HV_MRCO_GAUGES, _LV_MRCO_GAUGES, _HV_DREAM_GAUGES, _LV_DREAM_GAUGES]

# Pumps on the beamline
_BL_PUMP = vacDict['BL_PUMP']
 
# Pumps on LAMP
_LAMP_PUMP = vacDict['LAMP_PUMP']

# Pumps on MRCO
_MRCO_PUMP = vacDict['MRCO_PUMP']

# Pumps on DREAM
_DREAM_PUMP = ['DREAM:NC:PTM:01','DREAM:NC:PTM:02','DREAM:DP1:PTM:01','DREAM:DP2:PTM:01','DREAM:DP3:PTM:01']

# TMO Rouging Valves 
_TMO_Rough_Valves = vacDict['TMO_ROUGH_VALVES']

# LAMP Roughing Valves
_LV_LAMP_VALVES = vacDict['LV_LAMP_VALVES']

# MRCO Roughing Valves
_LV_MRCO_VALVES = vacDict['LV_MRCO_VALVES']

# Ion Pumps
#_ION_PUMP = ['MR1K4:SOMS:PIP:01','TV1K4:PIP:01','ST1K4:TEST:PIP:01','ST2K4:BCS:PIP:01','ST3K4:PPS:PIP:01','PC4K4:XTES:PIP:01','MR2K4:KBO:PIP:01','PA2K4:IP1:PIP:01','MR4K4:KBO:PIP:01']
_ION_PUMP = vacDict['ION_PUMP']

Gauge_States = ['Off','Disconnected','OoR','PressInvalid','Starting','On','ValidHi','ValidLo']
Pump_States = ['STOPPED','STARTING', 'RUNNING', 'FAULT', 'STOPPING']
Valve_States=['OPEN','CLOSED','MOVING','INVALID','OPEN_F']

    

def _build_dict(PUMP_LIST, GAUGE_LIST):
    defn = OrderedDict()
    for pump in PUMP_LIST:
        name = pump.replace(':','_')
        defn[name.lower()]=Cpt(PTMPLC, pump, kind='hinted')
    for gauge in GAUGE_LIST:
        name = gauge.replace(':','_')
        defn[name.lower()]=Cpt(GCCPLC, gauge, kind='hinted')
    return defn

class TmoHvacBase(BaseInterface, Device):
    """
    This is a class for all of the high vacuum devices (gauges and pumps) associated with the IP.
    Methods:
    --------------------
    vac_status: displays a dataframe of all the HV gauge pressures.
    pump_status: displays a dataframe of all the pump currents and speeds. 
    """
    tab_component_names= True
    tab_whitelist = ['vac_status', 'pump_status']
          
    def vac_status(self):
        """Get the status (Pressure and at_Vac boolean) of gauges on the endstation. """
        gauge_key = dir(self)#.__dict__.keys()
        state_dict = OrderedDict()
        pressure_dict = OrderedDict()
        status_dict = OrderedDict()
        for key in gauge_key:
            gauge = getattr(self, key)
            if hasattr(gauge, 'pressure'):
                if gauge.connected:
                    pressure_dict[key] = gauge.pressure.get()*1E8
                    status_dict[key] = bool(gauge.gauge_at_vac.get())
                    state_dict[key] = Gauge_States[gauge.state.get()]
                else:
                    state_dict[key] = 'n/c'
                    pressure_dict[key] = 'n/c'
                    status_dict[key] = 'n/c'
        gaugeDF = pd.DataFrame({"Gauge:": pressure_dict.keys(), "State": state_dict.values(), "Pressure (1E-8):": pressure_dict.values(), "at_Vac:": status_dict.values()})
        print (gaugeDF)
            
    def pump_status(self):
        """Returns the status of the pumps: at_speed information and the pump current for the pumps."""
        pump_key = dir(self)#.__dict__.keys()
        current_dict = OrderedDict()
        status_dict = OrderedDict()
        speed_dict = OrderedDict()
        for key in pump_key:
            pump = getattr(self, key)
            if hasattr(pump, 'pump_at_speed'):
                if pump.connected:
                    current_dict[key] = pump.pump_current.get()
                    status_dict[key] = Pump_States[pump.pump_state.get()] 
                    speed_dict[key] = bool(pump.pump_at_speed.get())
                else:
                    current_dict[key] = 'n/c'
                    status_dict[key] = 'n/c'
                    speed_dict[key] = 'n/c'
        pumpDF = pd.DataFrame({"Pump:": current_dict.keys(), "Pump Current:": current_dict.values(), "at_Speed:": speed_dict.values(), "Pump_State:": status_dict.values()})
        print (pumpDF)

class TmoHVacBl(TmoHvacBase):
    """
    This is a class for all of the high vacuum devices (gauges, pumps, gate valves) along the TMO beamline, exclusive of the IP.
    Methods:
    --------------------
    vac_status  - retruns a dataframe of all the HV gauge pressures.
    pump_status - returns a dataframe of all the pump currents and speeds. 
    valve_status- get status of TMO vacuum gate valves
    open_valves - open all TMO valves
    close_valves- close all TMO valves
    """    
    
    tab_component_names= True
    tab_whitelist = ['valve_status','open_valves', 'close_valves']
    
    _fee_vac_valves=[tv1k4_vgc_01,tv1k4_vfs_01,tv1k4_vgc_02]
    _tmo_vac_valves=[st3k4_pps_vgc_01,mr2k4_kbo_vwc_01,mr3k4_kbo_vwc_01,mr3k4_kbo_vfs_01,li3k4_ip1_vwc_01,pc5k4_vwc_01]#,mr4k4_kbo_vwc_01,mr5k4_kbo_vwc_01,tm2k4_kbo_vwc_01]  
    
    def valve_status(self):
        """Get the status of the vacuum valves along the TMO beamline."""
        ilk_state = {}
        curr_state= {}
        for ind, valve in enumerate((self._fee_vac_valves+self._tmo_vac_valves)):
            curr_state[valve.name] = Valve_States[valve.valve_position.get()]
            if hasattr(valve, 'interlock_ok'):
                ilk_state[valve.name] = bool(valve.interlock_ok.get())   
            elif hasattr(valve, 'vac_fault_ok'):
                ilk_state[valve.name]= bool(valve.vac_fault_ok.get())
            else:
                ilk_state[valve.name] = 'n/a'
        valveDF = pd.DataFrame({"Valve:": curr_state.keys(), "Position:": curr_state.values(), "Interlock ok:": ilk_state.values()})
        print (valveDF)
    
    def open_valves(self):
        """Open all vacuum gate valves on TMO beamline from ST3K4 thru TM2K4"""
        for ind, valve in enumerate(self._tmo_vac_valves):
            if hasattr(valve, 'open_command'):
                if bool(valve.interlock_ok.get()):
                    valve.open_command.put(1)
                else: print(valve.name+" interlock is not ok.")

        self.valve_status()

    def close_valves(self):
        """Close all vacuum gate valves on TMO beamline from ST3K4 thru TM2K4"""
        for ind, valve in enumerate(self._tmo_vac_valves):
            if hasattr(valve, 'open_command'):
                valve.open_command.put(0)

        self.valve_status()

def _build_rough_dict(GAUGE_LIST,VALVE_LIST):
    defn=OrderedDict()
    for gauge in GAUGE_LIST:
        name = gauge.replace(':','_')
        defn[name.lower()] = Cpt(GaugePLC, gauge, kind='hinted')
    for valve in VALVE_LIST:
        name = valve.replace(':','_')
        valve_type = re.search(':V..:', valve)
        if (valve_type.group()[1:-1] == 'VRC'):
            defn[name.lower()] = Cpt(VGC_2S, valve, kind='hinted')
        elif (valve_type.group()[1:-1] == 'VFV'):
            defn[name.lower()] = Cpt(VCN, valve, kind='hinted')
                
    return defn

class TmoRoughBase(BaseInterface, Device):
    """
    """
    tab_component_names= True
    tab_whitelist = ['status']
    
    def status(self):
                
        ilk_state = OrderedDict()
        curr_state= OrderedDict()
        device_keys = dir(self)#.__dict__.keys()
        
        for key in device_keys:
            device = getattr(self, key) 
            if hasattr(device, 'valve_position'):
                if device.connected:
                    curr_state[key] = Valve_States[device.valve_position.get()]
                    ilk_state[key] = bool(device.interlock_ok.get())
                else:
                    curr_state[key] = 'n/c'
                    ilk_state[key] = 'n/c'
            elif hasattr(device, 'pressure'):
                if device.connected:
                    curr_state[key] = device.pressure.get() * 1E3
                    ilk_state[key] = bool(device.pressure_ok.get())
                else:
                    curr_state[key] = 'n/c'
                    ilk_state[key] = 'n/c'
     
        valveDF= pd.DataFrame({"Device:": curr_state.keys(), "Current State:": curr_state.values(), "Interlock:": ilk_state.values()})
        print (valveDF)
 
def _build_ion_dict(ION_PUMP):
    defn=OrderedDict()
    for pump in ION_PUMP:
        name = pump.replace(':','_')
        defn[name.lower()] = Cpt(PIPPLC, pump, kind='hinted')
    return defn   
    
class TmoIonPump(BaseInterface, Device):
    '''
    TMO Ion Pumps
    ---------------
    status: return the status of Ion Pumps in TMO.
    all_on: Start all TMO Ion Pumps.
    '''
    
    tab_component_names= True
    tab_whitelist = ['status','all_on']

    def status(self):
        """
        TMO Ion pump status
        """
        press_dict = OrderedDict()
        curr_state= OrderedDict()
        device_keys = dir(self)#.__dict__.keys()
        state = ['OFF', 'STARTING','ON','FAULT','STOPING']
        for key in device_keys:
            device = getattr(self, key)
            if hasattr(device, 'pump_on_status'):
                if device.connected:
                    curr_state[key] = state[device.pump_state.get()]
                    press_dict[key] = device.pressure.get()*1E10
                else:
                    curr_state[key] = 'n/c'
                    press_dict[key] = 'n/c'

        pumpDF = pd.DataFrame({"Pump:": curr_state.keys(), "Current State:": curr_state.values(), "Pressure (1E-10):": press_dict.values()})
        print (pumpDF) 
        
    def all_on(self):
        """
        Start all TMO Ion pumps            
        -------
        None.
        
        """
        pump_list = dir(self)
        for pump_name in pump_list:
            pump = getattr(self, pump_name)
            if hasattr(pump, 'high_voltage_switch') :
                if bool(pump.connected):
                    if bool(pump.interlock_ok.get()):
                        pump.high_voltage_switch.put(1)
                    else:
                        print(pump_name, ' interlock is not ok')
                else:
                    print(pump_name, ' is not connected.')
 
TmoHvac = type('TmoHvac',(TmoHVacBl, BaseInterface, Device), _build_dict(_BL_PUMP, _HV_BL_GAUGES))
tmo_hvac = TmoHvac(name='tmo_hvac')

TmoLamp = type('TmoLamp',(TmoHvacBase, BaseInterface, Device), _build_dict(_LAMP_PUMP, _HV_LAMP_GAUGES))
tmo_hvac_lamp = TmoLamp(name='tmo_hvac_lamp')

TmoMrco = type('TmoMrco',(TmoHvacBase, BaseInterface, Device), _build_dict(_MRCO_PUMP, _HV_MRCO_GAUGES))
tmo_hvac_mrco = TmoMrco(name='tmo_hvac_mrco')

TmoDream = type('TmoDream',(TmoHvacBase, BaseInterface, Device), _build_dict(_DREAM_PUMP, _HV_DREAM_GAUGES))
tmo_hvac_dream = TmoDream(name='tmo_hvac_dream')

TmoIon = type('TmoIon', (TmoIonPump, BaseInterface, Device), _build_ion_dict(_ION_PUMP))
tmo_pip = TmoIon(name='tmo_pip')

TmoRough = type('TmoRough', (TmoRoughBase, BaseInterface, Device), _build_rough_dict(_LV_BL_GAUGES,_TMO_Rough_Valves))
tmo_rough = TmoRough(name='tmo_rough')

TmoLampRough = type('TmoRough', (TmoRoughBase, BaseInterface, Device), _build_rough_dict(_LV_LAMP_GAUGES, _LV_LAMP_VALVES)) 
tmo_lamp_rough = TmoLampRough(name='tmo_lamp_rough')

TmoMrcoRough = type('TmoRough', (TmoRoughBase, BaseInterface, Device), _build_rough_dict(_LV_MRCO_GAUGES, _LV_MRCO_VALVES))
tmo_mrco_rough = TmoMrcoRough(name='tmo_mrco_rough')
