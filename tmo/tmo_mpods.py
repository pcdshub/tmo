from pcdsdevices.mpod_apalis import MPODApalisChannel, MPODApalisModule, MPODApalisModule4Channel, MPODApalisModule8Channel, MPODApalisModule16Channel, MPODApalisModule24Channel
from ophyd.device import Component as Cpt
from ophyd import EpicsSignal
from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from collections import OrderedDict
#from elog import HutchELog as elog_temp
#elog = elog_temp.from_registry()
from tmo.db import elog
import pandas as pd
import time

#dicts for all channels
mpod01_m0 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:01:M0:C%d'%i, name='mpod01_m0_c%d'%i) for i in range(24)}
mpod01_m1 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:01:M1:C%d'%i, name='mpod01_m1_c%d'%i) for i in range(24)}
mpod01_m2 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:01:M2:C%d'%i, name='mpod01_m2_c%d'%i) for i in range(24)}
mpod01_m3 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:01:M3:C%d'%i, name='mpod01_m3_c%d'%i) for i in range(24)}
mpod01_m4 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:01:M4:C%d'%i, name='mpod01_m4_c%d'%i) for i in range(16)}
mpod01_m5 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:01:M5:C%d'%i, name='mpod01_m5_c%d'%i) for i in range(16)}
mpod01_m6 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:01:M6:C%d'%i, name='mpod01_m6_c%d'%i) for i in range(16)}
mpod01_m7 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:01:M7:C%d'%i, name='mpod01_m7_c%d'%i) for i in range(16)}
mpod01_m8 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:01:M8:C%d'%i, name='mpod01_m8_c%d'%i) for i in range(16)}
mpod01_m9 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:01:M9:C%d'%i, name='mpod01_m9_c%d'%i) for i in range(16)}

mpod02_m0 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:02:M0:C%d'%i, name='mpod02_m0_c%d'%i) for i in range(8)}
mpod02_m1 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:02:M1:C%d'%i, name='mpod02_m1_c%d'%i) for i in range(12)}
mpod02_m2 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:02:M2:C%d'%i, name='mpod02_m2_c%d'%i) for i in range(8)}
mpod02_m3 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:02:M3:C%d'%i, name='mpod02_m3_c%d'%i) for i in range(8)}
mpod02_m4 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:02:M4:C%d'%i, name='mpod02_m4_c%d'%i) for i in range(12)}
mpod02_m5 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:02:M5:C%d'%i, name='mpod02_m5_c%d'%i) for i in range(4)}
mpod02_m6 = { 'c%d'%i: MPODApalisChannel('TMO:MPOD:02:M6:C%d'%i, name='mpod02_m6_c%d'%i) for i in range(4)}

mpod01_mod = { 'm%d'%i: MPODApalisModule('TMO:MPOD:01:M%d'%i, name='mpod01_m%d'%i) for i in range(10)}
mpod02_mod = { 'm%d'%i: MPODApalisModule('TMO:MPOD:02:M%d'%i, name='mpod02_m%d'%i) for i in range(7)}

mpod_list = [mpod01_m1, mpod01_m2, mpod01_m3, mpod01_m4, mpod01_m5, mpod01_m6, mpod01_m7, mpod01_m8, mpod01_m9, mpod02_m1, mpod02_m2, mpod02_m3, mpod02_m4, mpod02_m5, mpod02_m6]

    
class MpodMod(BaseInterface, Device):
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
    
    mpod02_m0 = Cpt(MPODApalisModule8Channel, ':02:M0')
    mpod02_m1 = Cpt(MPODApalisModule8Channel, ':02:M1')
    mpod02_m2 = Cpt(MPODApalisModule8Channel, ':02:M2')
    mpod02_m3 = Cpt(MPODApalisModule8Channel, ':02:M3')
    mpod02_m4 = Cpt(MPODApalisModule8Channel, ':02:M4')
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
        mpod_voltage_dict=OrderedDict()
        mpod_current_dict=OrderedDict()
        mpod_fault_dict=OrderedDict()
        
        keys = dir(self)
        for name in keys:
            mpod_mod = getattr(self, name)
            if hasattr(mpod_mod, 'faults'):
                mpod_voltage_dict[name] = mpod_mod.voltage_ramp_speed.get()
                mpod_current_dict[name] = mpod_mod.current_ramp_speed.get()
                mpod_fault_dict[name] = mpod_mod.faults.get()
        mpodDF = pd.DataFrame({"mpod_mod:": mpod_voltage_dict.keys(), 
                                 "Faults:":mpod_fault_dict.values(), 
                                 "Voltage Ramp Speed:": mpod_voltage_dict.values(), 
                                 "Current Ramp Speed:": mpod_current_dict.values()})
        print (mpodDF)

    def _find_named_channels(self):
        defn=OrderedDict() 
        for mpod_key in dir(self):
            mpod = getattr(self, mpod_key)
            if (hasattr(mpod, 'faults')):# and mpod.connected):
                for ch_key in dir(mpod):
                    ch = getattr(mpod, ch_key)
                    if (hasattr(ch, 'desc')):# and ch.connected):
                        ch_name = ch.desc.get().replace(' ','_')
                        if ( (ch_name!='') and (ch_name.lower()!='dead') and (ch_name.isidentifier()) ):
                            pv = ch.desc.pvname[0:17]
                            if ch.desc.pvname[17] != ':':
                                pv=pv + ch.desc.pvname[17]
                            defn[ch_name] = Cpt(MPODApalisChannel, pv, kind='hinted')
    #                        print(ch_name + ' found')
        return defn
        
mpod_modules = MpodMod('TMO:MPOD', name='tmo_mpod_mods')
 
class MpodCl(BaseInterface, Device):
    tab_component_names = True
    tab_whitelist = ['clear_faults_all','voltage_status','all_off']
    
    mpod_modules = Cpt(MpodMod, 'TMO:MPOD', kind = 'omitted')
    
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
        mpod_voltage_dict=OrderedDict()
        mpod_set_dict=OrderedDict()
        mpod_ch_dict=OrderedDict()
        mpod_state_dict=OrderedDict()
        
        mpod_key = dir(self)
        for key in mpod_key:
            mpod = getattr(tmo_mpod, key)
            if hasattr(mpod, 'voltage'):
                mpod_voltage_dict[key] = mpod.voltage.get()
                mpod_set_dict[key] = mpod.voltage_setpoint.get()
                mpod_state_dict[key] = mpod.state.get()
                mpod_ch_dict[key] = mpod.voltage.pvname[4:17]
                if mpod.voltage.pvname[17] != ':':
                    mpod_ch_dict[key] = mpod_ch_dict[key] + mpod.voltage.pvname[17]
        chDF = pd.DataFrame({"Channel:": mpod_voltage_dict.keys(), 
                             "State:": mpod_state_dict.values(),
                             "Voltage Set (V):": mpod_set_dict.values(), 
                             "Voltage Measure (V):": mpod_voltage_dict.values(), 
                             "Supply:":mpod_ch_dict.values()})
        sorted_posDF = chDF.sort_values(by=["Channel:"], ascending=True)
        print (sorted_posDF)

        if (log is True):
            if (msg is not None):
                msg = msg + "\n" + sorted_posDF.to_string()
            else:
                msg = sorted_posDF.to_string()
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
    
# def _build_mpod_dict():
#     #'TMO:MPOD:01:M0:C%d'
#     time.sleep(0.1)
#     defn=OrderedDict()
#     desc_dict = OrderedDict()
#     for mpod_crate in range(1,3):
#         crate_pv_str = 'TMO:MPOD:0%d'%mpod_crate
#         for mpod_mod in range(10):
#             mod_pv_str = crate_pv_str + ':M%d'%mpod_mod
#             for channel in range(24):
#                 channel_pv_str = mod_pv_str + ':C%d'%channel
#                 desc_pv = channel_pv_str+':VoltageMeasure.DESC'
#                 desc_dict[channel_pv_str.replace(':', '_')] = EpicsSignal( desc_pv, string=True, connection_timeout=1)
#     time.sleep(5)
#     for key in desc_dict.keys():
#         desc = desc_dict[key]
#         if bool(desc.connected):
#             desc_str = desc.get().replace(' ','_')
#             if ( (desc_str!='') and (desc_str.lower()!='dead') and (desc_str.isidentifier()) ):
#                  defn[desc_str] = Cpt(MPODApalisChannel, key.replace('_',':'), kind='hinted')
#     return defn

def _build_mpod_dict2():
#    mpod_modules = MpodMod('TMO:MPOD', name='tmo_mpod_mods')
    time.sleep(0.5)
    defn=OrderedDict() 
    for mpod_key in dir(mpod_modules):
        mpod = getattr(mpod_modules, mpod_key)
        if (hasattr(mpod, 'faults')):# and mpod.connected):
            for ch_key in dir(mpod):
                ch = getattr(mpod, ch_key)
                if (hasattr(ch, 'desc')):# and ch.connected):
                    ch_name = ch.desc.get().replace(' ','_')
                    if ( (ch_name!='') and (ch_name.lower()!='dead') and (ch_name.isidentifier()) ):
                        pv = ch.desc.pvname[0:17]
                        if ch.desc.pvname[17] != ':':
                            pv=pv + ch.desc.pvname[17]
                        defn[ch_name] = Cpt(MPODApalisChannel, pv, kind='hinted')
#                        print(ch_name + ' found')
    return defn
           
#TmoMpod = type('TmoMpod',(MpodCl, BaseInterface, Device), _build_mpod_dict2())
TmoMpod = type('TmoMpod',(MpodCl, BaseInterface, Device), mpod_modules._find_named_channels() )
tmo_mpod = TmoMpod(name='tmo_mpod')

def get_mpod(log=False, msg=None, **kwargs):
    """
    Get TMO MPOD Status. Optional to save to elog with a msg. 

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