from pcdsdevices.mpod_apalis import MPODApalisChannel, MPODApalisModule, MPODApalisModule4Channel, MPODApalisModule8Channel, MPODApalisModule16Channel, MPODApalisModule24Channel
from ophyd.device import Component as Cpt
from ophyd import EpicsSignal, EpicsSignalRO
from pcdsdevices.device import Device, GroupDevice
from pcdsdevices.interface import BaseInterface
from collections import OrderedDict
#from elog import HutchELog as elog_temp
#elog = elog_temp.from_registry()
from tmo.db import elog
import pandas as pd
from tabulate import tabulate
import time
import json
import logging
logger = logging.getLogger(__name__)


#class MPODApalisChannel_fix(MPODApalisChannel):
#    voltage = Cpt(EpicsSignal, ':VoltageMeasure', write_pv=':VoltageSet', kind='normal')
#    event_trip = Cpt(EpicsSignalRO, ':EventTrip', kind='normal')
#    def set_voltage(self, voltage_number: float) -> None:
#        max_voltage = self.max_voltage.get()

#        if abs(voltage_number) <= abs(max_voltage):
#            self.voltage.put(voltage_number)
#        else:
#            logger.error(("%s is out of range for %s" % (voltage_number, self.name)))
#            #self.voltage.put(max_voltage)
    
    
  
class MpodMod(BaseInterface, Device):
    """Class for all the TMO Mpod Modules."""
    
    tab_component_names = True
    tab_whitelist = ['clear_faults_all', 'status']
    
    mpod01_m0 = Cpt(MPODApalisModule24Channel, ':01:M0')
    mpod01_m1 = Cpt(MPODApalisModule24Channel, ':01:M1')
    mpod01_m2 = Cpt(MPODApalisModule24Channel, ':01:M2')
    mpod01_m3 = Cpt(MPODApalisModule24Channel, ':01:M3')
    mpod01_m4 = Cpt(MPODApalisModule16Channel, ':01:M4')
    mpod01_m5 = Cpt(MPODApalisModule16Channel, ':01:M5')
    mpod01_m6 = Cpt(MPODApalisModule16Channel, ':01:M6')
    mpod01_m7 = Cpt(MPODApalisModule16Channel, ':01:M7')
    mpod01_m8 = Cpt(MPODApalisModule16Channel, ':01:M8')
    mpod01_m9 = Cpt(MPODApalisModule16Channel, ':01:M9')
    
    mpod02_m0 = Cpt(MPODApalisModule24Channel, ':02:M0')
    mpod02_m1 = Cpt(MPODApalisModule24Channel, ':02:M1')
    mpod02_m2 = Cpt(MPODApalisModule24Channel, ':02:M2')
    mpod02_m3 = Cpt(MPODApalisModule16Channel, ':02:M3')
    mpod02_m4 = Cpt(MPODApalisModule16Channel, ':02:M4')
    mpod02_m5 = Cpt(MPODApalisModule4Channel, ':02:M5')
    mpod02_m6 = Cpt(MPODApalisModule4Channel, ':02:M6')
    
    def clear_faults_all(self):
        """
        Clear faults on all mpod modules. 
        """
        keys = dir(self)
        for name in keys:
            mpod_mod = getattr(self, name)
            if hasattr(mpod_mod, 'faults'):
                mpod_mod.faults.put(1)
        time.sleep(5)
        for name in keys:
            mpod_mod = getattr(self, name)
            if hasattr(mpod_mod, 'faults'):
                mpod_mod.faults.put(0)

    def status(self):
        """
        Get MPod module status
        """
        
        mpod_dict = {"mpod_mod:": [], 
                     "Faults:":[], 
                     "Voltage Ramp Speed:": [], 
                     "Current Ramp Speed:": []}
               
        keys = dir(self)
        for name in keys:
            mpod_mod = getattr(self, name)
            if hasattr(mpod_mod, 'faults'):
                mpod_dict["mpod_mod:"].append(name)
                mpod_dict["Voltage Ramp Speed:"].append(mpod_mod.voltage_ramp_speed.get())
                mpod_dict["Current Ramp Speed:"].append(mpod_mod.current_ramp_speed.get())
                mpod_dict["Faults:"].append(mpod_mod.faults.get())
        mpodDF = pd.DataFrame(mpod_dict)
        mpodDF_str = tabulate(mpodDF, headers="keys", tablefmt="pretty")
#        print (mpodDF)
        print (mpodDF_str)
    def _find_named_channels(self, timeout=10):
        """Find all the named channels for the mpod modules."""     
        defn=OrderedDict() 
        for mpod_key in dir(self):
            mpod = getattr(self, mpod_key)
            if (hasattr(mpod, 'faults')):# and mpod.connected):
                try: 
                    mpod.wait_for_connection(timeout=timeout)
                except TimeoutError:
                    logger.error(mpod.name + ' not connected')
                    continue
                for ch_key in dir(mpod):
                    ch = getattr(mpod, ch_key)
                    if (hasattr(ch, 'desc')):
                        ch.wait_for_connection(timeout=timeout)
                        ch.max_voltage.wait_for_connection(timeout=timeout)
                        if ch.connected and ch.max_voltage.connected:
                            ch_name = ch.desc.get().replace(' ','_')
                            if ( (ch_name!='') and (ch_name.lower()!='dead') and (ch_name.isidentifier()) ):
                                pv = ':'.join(ch.desc.pvname.split(':')[0:-1])
                                defn[ch_name] = Cpt(MPODApalisChannel, pv, kind='hinted')
                                defn[ch_name].biological_parent = ch.biological_parent
                                logger.debug(ch_name + ' found')
                        else:
                            logger.info(ch.name + ' not connected.')

        return defn
        
mpod_modules = MpodMod('TMO:MPOD', name='tmo_mpod_mods')
 
class MpodCl(BaseInterface, Device):
    """Class for interacting with named mpod modules."""
    tab_component_names = True
    tab_whitelist = ['clear_faults_all','voltage_status','all_off']
    
    mpod_modules = Cpt(MpodMod, 'TMO:MPOD', kind = 'omitted')
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        for ch_name, o_channel in self.mpod_modules._find_named_channels(0.5).items():
            ch = getattr(self, ch_name)
            ch.biological_parent = o_channel.biological_parent
            ch._new_max_voltage(ch.max_voltage.get())
            ch._update_max_voltage_limits()
            logger.debug('update voltage limits for %s' % ch_name)
        
    def clear_faults_all(self):
        """
        Clear faults on all mpod modules
        """
        self.mpod_modules.clear_faults_all()    
        
    def voltage_status(self, log=False, msg=None, **kwargs):
        """
        Get TMO MPOD Status. Optional to save to elog with a msg. 

        Parameters
        ----------
        log : Boolean, optional
            Post to elog. The default is False.
        msg : str, optional
            Message to attach to elog post. The default is None.
        run : int, optional
            Run number to append to elog post
        """
       
        mpod_dict={"Channel:": [], 
                   "State:": [],
                   "Voltage Set (V):": [], 
                   "Voltage Measure (V):":[],
                   "Current Measure (uA):":[],
                   "isTrip:":[],
                   "wasTrip:":[],
                   "Supply:":[]}
        
        mpod_key = dir(self)
        for key in mpod_key:
            mpod = getattr(self, key)
            if hasattr(mpod, 'voltage'):
                mpod_dict["Channel:"].append(key)
                mpod_dict["Voltage Measure (V):"].append(mpod.voltage.get())
                mpod_dict["Voltage Set (V):"].append(mpod.voltage_setpoint.get())
                mpod_dict["Current Measure (uA):"].append(mpod.current.get()*1E6)
                mpod_dict["State:"].append(mpod.state.get())
                mpod_dict["isTrip:"].append(bool(mpod.is_trip.get()))
                mpod_dict["wasTrip:"].append(bool(mpod.event_trip.get()))
#                supply_str = mpod.voltage.pvname[4:17]
#                if mpod.voltage.pvname[17] != ':':
#                    supply_str = supply_str + mpod.voltage.pvname[17]
#                mpod_dict["Supply:"].append(supply_str)
                mpod_dict["Supply:"].append(':'.join(mpod.voltage.pvname.split(':')[1:-1]))
        chDF = pd.DataFrame(mpod_dict)
        sorted_posDF = chDF.sort_values(by=["Channel:"], ascending=True)
        chDF_str = tabulate(sorted_posDF, headers="keys", tablefmt="pretty")
#        print (sorted_posDF)
        print(chDF_str)
        if (log is True):
            if (msg is not None):
#                msg = msg + "\n" + sorted_posDF.to_string()
                msg = msg + "\n" + chDF_str
            else:
#                msg = sorted_posDF.to_string()
                msg = chDF_str
            kwargs['tags'] = kwargs.get('tags'," ") + ' hv'
            elog.post(msg, **kwargs) 

    def all_off(self):
        """
        Turn off all named MPOD channels
        """
        mpod_key = dir(self)
        for key in mpod_key:
            mpod = getattr(tmo_mpod, key)
            if hasattr(mpod, 'off'):
                mpod.off()
    
           
#TmoMpod = type('TmoMpod',(MpodCl, BaseInterface, Device), _build_mpod_dict2())
# TmoMpod = type('TmoMpod',(MpodCl, BaseInterface, Device), mpod_modules._find_named_channels() )
TmoMpod = type('TmoMpod',(MpodCl, BaseInterface, GroupDevice), mpod_modules._find_named_channels() )
tmo_mpod = TmoMpod(name='tmo_mpod')

# author: roberttk 5/30/25 (also the GroupDevice change above)
# Re-assign the named channels their original parents.
# Once we have done this, refresh their max voltage limits.  
# We originally set this to auto update based on the parent module, but this
# new parent doesn't know it has an additional child.  So if we want to re-read
# the limits we'll have to update the max voltage and re-read the parent limits
#for channel_name, original_channel in mpod_modules._find_named_channels().items():
#    channel = getattr(tmo_mpod, channel_name)
#    channel.biological_parent = original_channel.biological_parent
#    channel._new_max_voltage(channel.max_voltage.get())
#    channel._update_max_voltage_limits()


def get_mpod(log=False, msg=None, **kwargs):
    """
    Get TMO MPOD Status. Optional to save to elog with a msg. 
;
    Parameters
    ----------
    log : Boolean, optional
        Post to elog. The default is False.
    msg : Boolean, optional
        Message to attach to elog post. The default is None.
    run : Boolean, optional
        Run number to append to elog post
    """
    tmo_mpod.voltage_status(log,msg,**kwargs)

def tmo_mpod_off():
    """
    Turn off all mpod channels
    """
    tmo_mpod.all_off()
    
def hv_checkout(mod = None):
    """
    Function to check out MPOD channels. This function can check out all modules or a specific module.
    For a specific module pass the modules as the optional mod arguement.
    The function prints any faults found

    Parameters
    ----------
    mod : MPOD_module, optional
        To check a specific module specify pass the module as this arguement. The default is None.

    Returns
    -------
    None.

    """
    def _check_mod(mod):
        if mod.faults.get() == 'Active event(s)':
            print(mod.name + ' has faults!')
        else:
            for chs in dir(mod):
                ch = getattr(mod, chs)
                if hasattr(ch, 'voltage_setpoint'):
                    ch.set_voltage(10)
                    ch.on()
            time.sleep(10)
            for chs in dir(mod):
                ch = getattr(mod, chs)
                if hasattr(ch, 'voltage_setpoint'):
                    Vr = (abs(ch.voltage_setpoint.get()) - abs(ch.voltage.get()))/ch.voltage_setpoint.get()                    
                    if abs(Vr) > 0.1:                          
                        print (ch.name + ' failed to ramp!')
                    ch.off()

    if mod is not None:
        _check_mod(mod)
    else:
        for modules in  dir(tmo_mpod.mpod_modules):
            mod = getattr(tmo_mpod.mpod_modules, modules)
            if hasattr(mod, 'faults'):
                _check_mod(mod)
                        
def mpod_clear_names(filename='/cds/home/opr/tmoopr/hutch-python/tmo/tmo_mpod_configs/old_names.json'):
    """
    This function cycles thru all of the channels on MPOD01 and MPOD02 and clears the description field. 

    """
    logger.info('Removing old mpod names.....')
    old_mpod_names = OrderedDict()
    for mpod_name in dir(tmo_mpod.mpod_modules):
        mpod = getattr(tmo_mpod.mpod_modules, mpod_name)
        if not ('MPOD' in str(type(mpod))): continue
        for ch_name in dir(mpod):
            ch = getattr(mpod, ch_name)
            if not ('MPOD' in str(type(ch))): continue
            current_name = ch.desc.get()
            ch.desc.put('')
            if not (current_name == ''):
                old_mpod_names[current_name] = ':'.join(ch.desc.pvname.split(':')[0:-1])
    
    with open(filename, 'w') as f:
        nameDict = json.dump(old_mpod_names, f, indent=4)
        logger.info('write old mpod names to %s....' % filename)
    return

import subprocess
from datetime import date
            
def mpod_name_from_file(filename):
    """
    Update the named channels for the TMO Mpods based on a dictionary saved in a .json file. The previous names are removed and saved in the old_names.json file.

    Parameters
    ----------
    filename : str
        Path to .json file containing a dictionary. The form of the dictionary should be keys that will be used as names and values for the specific channel PV.

    Returns
    -------
    None.

    """
    
    with open(filename, 'r') as f:
        nameDict = json.load(f)
    
    mpod_clear_names()
    
    epicsArch_entry = "# {'epicsArch_mpod01.txt': { \n# 'TITLE': 'Mpod HV PVs', \n# 'Updated': '%s', \n# 'Description': 'PVs related to high voltage Mpod. Update with mpod_name_from_file.' \n# }}\n\n" % date.today().strftime("%Y%m%d") 
        
    logger.info('adding names from %s...' % filename)
    for ch_name in nameDict.keys():
        pv = nameDict[ch_name]
        mpod_mod = nameDict[ch_name].split(":")[2]
        mpod_num = nameDict[ch_name].split(":")[3][1:]
        ch_num = nameDict[ch_name].split(":")[4][1:]
        
        mpod = getattr(tmo_mpod.mpod_modules, "mpod%s_m%s" % (mpod_mod, mpod_num))
        ch = getattr(mpod, "c%s" % (ch_num))
        ch.desc.put(ch_name)        
        epicsArch_entry += '*%s \n%s:VoltageMeasure ca \n\n' % (ch_name.replace(' ','_'), pv)
        
    #update desc to arch
    logger.info('Updating epicsArch_mpod01.txt....')
#    with open('/cds/home/opr/tmoopr/misc/epicsArch_mpod01.txt', 'w') as file:
    with open('/sdf/home/t/tmoopr/daq/epics/epicsArch_mpod01.txt', 'w') as file:
        file.write(epicsArch_entry)  
######Below here is an old way that used mpod_desc_to_arch.##########    
#    logger.info('running mpod_desc_to_arch...')
#    result = subprocess.run(['rm','/cds/home/opr/tmoopr/misc/epicsArch_mpod01.txt'], capture_output=True)
#    print (result.stdout)
#    result = subprocess.run(['mpod_desc_to_arch'], capture_output=True, text=True)
#    print (result.stdout)
    return
        
        

            
    
    
    
    
