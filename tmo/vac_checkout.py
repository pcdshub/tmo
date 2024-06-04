from ophyd import EpicsSignal, EpicsSignalRO
from pcdsdevices.gauge import GCCPLC, GaugePLC
from pcdsdevices.pump import PTMPLC, PIPPLC
from pcdsdevices.valve import VGC_2S
from ophyd.device import Component as Cpt
from tmo.db import tv1k4_vgc_01, tv1k4_vfs_01, tv1k4_vgc_02, st3k4_pps_vgc_01, mr2k4_kbo_vwc_01, mr3k4_kbo_vwc_01, mr3k4_kbo_vfs_01, li3k4_ip1_vwc_01, pc5k4_vwc_01, mr4k4_kbo_vwc_01, mr5k4_kbo_vwc_01, tm2k4_kbo_vwc_01
from epics import caput
import pandas as pd
import logging
import json

logger = logging.getLogger(__name__)

class Colors:
    OK = '\033[92m'
    FAIL = '\033[91m'

# These are items that are yes/no
_GAUGE_BOOL = {
    ':HV_SW': 'ON',
    ':LOGGER': 'ON',
    ':STATE_RBV': 'Valid',
    ':AT_VAC_RBV': 'TRUE',
    ':PRESS_OK_RBV': 'ON',
}

# These are items that have thresholds
_GAUGE_STATE = {
    ':VAC_SP': 0.001,
    ':SP_HYS': 0.001,
    ':PRO_SP': 0.001,
}



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

def _pass_fail_test(base, test_dict):
    # Yes no test
    for k, v in test_dict.items():
        item = EpicsSignalRO(f'{base}{k}')
        value = item.get()
        if item.enum_strs[value] == v:
            logger.info(f'{Colors.OK}{item.name}: Pass')
        else:
            logger.error(f'{Colors.FAIL}{item.name}: Failed \
expected {v}, got {item.enum_strs[value]}')

def _thresh_test(base, test_dict):
    # Threshold tests
    for k, v in test_dict.items():
        item = EpicsSignalRO(f'{base}{k}')
        if round(item.get(), 6) == v:
            logger.info(f'{Colors.OK}{item.name}: Pass')
        else:
            logger.error(f'{Colors.FAIL}{item.name}: Failed \
expected {v}, got {item.get()}')

def _check_gauge(base):
    """Standard test for cold cathod gauge"""
    _pass_fail_test(base, _GAUGE_BOOL)
    _thresh_test(base, _GAUGE_STATE)

def tmo_vac_checkout():
    for gauge in _GAUGES:
        _check_gauge(gauge)

class TmoHvacIp:
    """
    This is a class for all of the high vacuum devices (gauges and pumps) associated with the IP.
    Methods:
    --------------------
    vac_status: displays a dataframe of all the HV gauge pressures.
    pump_status: displays a dataframe of all the pump currents and speeds. 
    """
    def vac_status(self):
        """Get the status (Pressure and at_Vac boolean) of gauges on the endstation. """
        gauge_key = self.__dict__.keys()
        pressure_dict = {}
        status_dict = {}
        for key in gauge_key:
            gauge = getattr(self, key)
            if hasattr(gauge, 'pressure'):
                if gauge.connected:
                    pressure_dict[key] = gauge.pressure.get()*1E8
                    status_dict[key] = bool(gauge.gauge_at_vac.get())
                else:
                    pressure_dict[key] = 'n/c'
                    status_dict[key] = 'n/c'
        temp = {"Gauge:": pressure_dict.keys(), "Pressure (1E-8):": pressure_dict.values(), "at_Vac:": status_dict.values()}
        gagDF = pd.DataFrame(temp)
        print (gagDF)
            
    def pump_status(self):
        """Returns the status of the pumps: at_speed information and the pump current for the endstation pumps."""
        pump_key = self.__dict__.keys()
        current_dict = {}
        status_dict = {}
        for key in pump_key:
            pump = getattr(self, key)
            if hasattr(pump, 'pump_at_speed'):
                if pump.connected:
                    current_dict[key] = pump.pump_current.get()
                    status_dict[key] = bool(pump.pump_at_speed.get()) 
        temp = {"Pump:": current_dict.keys(), "Pump Current:": current_dict.values(), "at_Speed:": status_dict.values()}
        pumpDF = pd.DataFrame(temp)
        print (pumpDF)


class TmoHvac:
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
    def __init__(self):
        self._fee_vac_valves=[tv1k4_vgc_01,tv1k4_vfs_01,tv1k4_vgc_02]
        self._tmo_vac_valves=[st3k4_pps_vgc_01,mr2k4_kbo_vwc_01,mr3k4_kbo_vwc_01,mr3k4_kbo_vfs_01,li3k4_ip1_vwc_01,pc5k4_vwc_01]#,mr4k4_kbo_vwc_01,mr5k4_kbo_vwc_01,tm2k4_kbo_vwc_01]  
                      #,mr4k4_kbo_vfs_01]
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
        temp = {"Valve:": curr_state.keys(), "Position:": curr_state.values(), "Interlock ok:": ilk_state.values()}
        posDF = pd.DataFrame(temp)
        print (posDF)
    
    def open_valves(self):
        """Open all vacuum gate valves on TMO beamline from ST3K4 thru TM2K4"""
        for ind, valve in enumerate(self._tmo_vac_valves):
            if hasattr(valve, 'open_command'):
                if bool(valve.interlock_ok.get()):
                    valve.open_command.put(1)
                else: print(valve.name+" interlock is not ok.")
#        caput("MR4K4:KBO:VWC:01:OPN_SW",1)
#        caput("MR5K4:KBO:VWC:01:OPN_SW",1)
        self.valve_status()

    def close_valves(self):
        """Close all vacuum gate valves on TMO beamline from ST3K4 thru TM2K4"""
        for ind, valve in enumerate(self._tmo_vac_valves):
            if hasattr(valve, 'open_command'):
                valve.open_command.put(0)
#        caput("MR4K4:KBO:VWC:01:OPN_SW",0)
#        caput("MR5K4:KBO:VWC:01:OPN_SW",0)
        self.valve_status()

    def vac_status(self):
        """Get the status (Pressure and at_Vac boolean) of gauges along the TMO beamline """
        gauge_key = self.__dict__.keys()
        pressure_dict = {}
        status_dict = {}
        for key in gauge_key:
            gauge = getattr(self, key)
            if hasattr(gauge, 'pressure'):
                if gauge.connected:
                    if (gauge.state.get() == 5):
                        pressure_dict[key] = gauge.pressure.get()*1E8
                        status_dict[key] = bool(gauge.gauge_at_vac.get())
                    else:
                        pressure_dict[key] = Gauge_States[gauge.state.get()]
                        status_dict[key] = bool(gauge.gauge_at_vac.get())
                else:
                    pressure_dict[key] = 'n/c'
                    status_dict[key] = 'n/c'
        temp = {"Gauge:": pressure_dict.keys(), "Pressure (1E-8):": pressure_dict.values(), "at_Vac:": status_dict.values()}
        gagDF = pd.DataFrame(temp)
        print (gagDF)
            
    def pump_status(self):
        """Returns the status of the pumps: at_speed information and the pump current along the TMO beamline"""
        pump_key = self.__dict__.keys()
        current_dict = {}
        status_dict = {}
        for key in pump_key:
            pump = getattr(self, key)
            if hasattr(pump, 'pump_at_speed'):
                if pump.connected:
                    current_dict[key] = pump.pump_current.get()
                    status_dict[key] = bool(pump.pump_at_speed.get()) 
        temp = {"Pump:": current_dict.keys(), "Pump Current:": current_dict.values(), "at_Speed:": status_dict.values()}
        pumpDF = pd.DataFrame(temp)
        print (pumpDF) 


def build_tmo_vac_ip(GAUGE_LIST, PUMP_LIST):
    tmo_hvac = TmoHvacIp()
    for gauge in GAUGE_LIST:
        name = gauge.replace(':','_')
        setattr(tmo_hvac, name.lower(), GCCPLC(gauge, name=name.lower()))
    
    for pump in PUMP_LIST:
        name = pump.replace(':','_')
        setattr(tmo_hvac, name.lower(), PTMPLC(pump, name=name.lower()))
    return tmo_hvac

def build_tmo_vac_bl(GAUGE_LIST, PUMP_LIST):
    tmo_hvac = TmoHvac()
    for gauge in GAUGE_LIST:
        name = gauge.replace(':','_')
        setattr(tmo_hvac, name.lower(), GCCPLC(gauge, name=name.lower()))
        
    for pump in PUMP_LIST:
        name = pump.replace(':','_')
        setattr(tmo_hvac, name.lower(), PTMPLC(pump, name=name.lower()))

    for ind, valve in enumerate(tmo_hvac._tmo_vac_valves+tmo_hvac._fee_vac_valves):
        name = valve.name
        setattr(tmo_hvac, name.lower(), valve)

    return tmo_hvac

tmo_hvac = build_tmo_vac_bl(_HV_BL_GAUGES, _BL_PUMP)
tmo_hvac_lamp = build_tmo_vac_ip(_HV_LAMP_GAUGES, _LAMP_PUMP)
tmo_hvac_mrco = build_tmo_vac_ip(_HV_MRCO_GAUGES, _MRCO_PUMP)
tmo_hvac_dream = build_tmo_vac_ip(_HV_DREAM_GAUGES, _DREAM_PUMP)

class TmoRough:
    def status(self):
        v_states=['OPEN','CLOSED','MOVING','INVALID','OPEN_F']
        
        ilk_state = {}
        curr_state= {}
        device_keys = self.__dict__.keys()
        
        for key in device_keys:
            device = getattr(self, key) 
            if ( hasattr(device, 'valve_position') and getattr(device, 'connected') ):
                curr_state[device.name] = v_states[device.valve_position.get()]
                ilk_state[device.name] = bool(device.interlock_ok.get())
            elif ( hasattr(device, 'pressure') and getattr(device, 'connected') ) :
                curr_state[device.name] = device.pressure.get() * 1E3
                ilk_state[device.name] = bool(device.pressure_ok.get())
      
        temp = {"Device:": curr_state.keys(), "Current State:": curr_state.values(), "Interlock:": ilk_state.values()}
        posDF = pd.DataFrame(temp)
        print (posDF)
        
def build_tmo_rough(GAUGE_LIST,VALVE_LIST):
    tmo_rough = TmoRough()
    for gauge in GAUGE_LIST:
        name = gauge.replace(':','_')
        setattr(tmo_rough, name.lower(), GaugePLC(gauge, name=name.lower()))
    for valve in VALVE_LIST:
        name = valve.replace(':','_')
        setattr(tmo_rough, name.lower(), VGC_2S(valve, name=name.lower()))
    
    return tmo_rough

tmo_rough = build_tmo_rough(_LV_BL_GAUGES,_TMO_Rough_Valves)
tmo_lamp_rough = build_tmo_rough(_LV_LAMP_GAUGES, _LV_LAMP_VALVES) 
tmo_mrco_rough = build_tmo_rough(_LV_MRCO_GAUGES, _LV_MRCO_VALVES) 

class TmoIonPump:
    '''
    TMO Ion Pumps
    ---------------
    status: 
    '''
    def status(self):
                
        press_dict = {}
        curr_state= {}
        device_keys = self.__dict__.keys()
        state = ['OFF', 'STARTING','ON','FAULT','STOPING']
        for key in device_keys:
            device = getattr(self, key)
            if hasattr(device, 'pump_on_status'):
                if device.connected:
                    curr_state[device.name] = state[device.pump_state.get()]
                    press_dict[device.name] = device.pressure.get()*1E10
                else:
                    curr_state[device.name] = 'n/c'
                    press_dict[device.name] = 'n/c'

        temp = {"Pump:": curr_state.keys(), "Current State:": curr_state.values(), "Pressure (1E-10):": press_dict.values()}
        pumpDF = pd.DataFrame(temp)
        print (pumpDF) 

def build_tmo_pip(ION_PUMP):
    tmo_pip=TmoIonPump()
    for pump in ION_PUMP:
        name = pump.replace(':','_')
        setattr(tmo_pip, name.lower(), PIPPLC(pump, name=name.lower()))
    return tmo_pip

tmo_pip = build_tmo_pip(_ION_PUMP)
