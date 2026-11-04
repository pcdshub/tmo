from ophyd.device import Component as Cpt
from ophyd.device import FormattedComponent as FCpt
#from ophyd.device import Staged
from pcdsdevices.device import UnrelatedComponent as UCpt
from pcdsdevices.device import GroupDevice, Device
from pcdsdevices.attenuator import GattApertureY
from pcdsdevices.interface import BaseInterface
from ophyd.signal import EpicsSignal, EpicsSignalRO
from pcdsdevices.pv_positioner import PVPositioner, PVPositionerIsClose
from ophyd.status import DeviceStatus
from pcdsdevices.valve import VVC, VCN
from tmo.db import tmo_cvmi, tmo_lamp, tmo_mrco, tv1k4_vgc_01, tv1k4_vfs_01, tv1k4_vgc_02, st3k4_pps_vgc_01, mr2k4_kbo_vwc_01, mr3k4_kbo_vwc_01, mr3k4_kbo_vfs_01, li3k4_ip1_vwc_01, pc5k4_vwc_01, rtdsk0
from tmo.db import im2k0, st1k0, st3k4_pps, st4k4, elog
from tmo.tmo_hvac import tmo_hvac
import matplotlib.pyplot as plt
import pyfiglet
import wave
import pyaudio
from tmo.tmo_utils import log_positions as _log_positions

import pandas as pd
import numpy as np
from collections import OrderedDict
import time
import sys
import h5py
import datetime
import logging
logger = logging.getLogger(__name__)

   
class TmoBeamLinePositioner(PVPositionerIsClose):
    setpoint = Cpt(EpicsSignal, '', kind='hinted')
    readback = Cpt(EpicsSignalRO, '_RBV', kind='normal')
    actuate = FCpt(EpicsSignal, 'PMPS:KFE:BeamParamCntl:ReqBP:Apply', kind='hinted')
    actuate_value = 1
    atol=0.1
    rtol=0.1

class PmpsOverride(BaseInterface, GroupDevice):
    """
    This class controls the PMPS override for the TMO beamline.
    Methods:
    ----------
        apply(max_eng, duration): apply the pmps override with max energy for the duration
        cancel() : Cancel the pmps override
        status(): Give the status of the PMPS override.
    """
    tab_component_names = True
    tab_whitelist = ['apply','cancel','status']
    
    max_eng = Cpt(EpicsSignal, ':IntensityJF_RBV', write_pv=':IntensityJF', kind = 'hinted',
                                doc='PMPS Override Max Credible Energy [mJ]')
    duration =  Cpt(EpicsSignal, ':DurationJF_RBV', write_pv=':DurationJF', kind = 'hinted',
                                  doc='PMPS Override Timeout Duration [Hrs]')
    time_remaining = Cpt(EpicsSignalRO, ':TimeRemainJF_RBV', kind='hinted',
                                       doc='PMPS Override Time Remaining [Hrs]')
    _apply_cmd = Cpt(EpicsSignal, ':ApplyJF_RBV', write_pv=':ApplyJF', kind='omitted')
    _cancel_cmd = Cpt(EpicsSignal, ':CancelJF', kind='omitted')
    _user_trans_request = Cpt(EpicsSignal, ':BeamParamCntl:ReqBP:Transmission_RBV', kind='omitted')
    _local_max_eng = FCpt(EpicsSignal, 'TMO:PMPS:MAX:ENERGY', kind='omitted',
                          doc='Notepad PV to keep track of the initial max energy request.')
    _facility_mode = FCpt(EpicsSignalRO, 'SIOC:FEES:MP01:FACMODE_RBV', string=True, kind='omitted',
                        doc='Facility Operating Mode [SC or NC]')
    
    
    def _trans_adjust(self, *args, value, timestamp, **kwargs):
        """ This function monitors the line beam parameter attenuation request,
        and adjusts the pmps beam power override based on this value. This will 
        make it possible to track beam class."""
        new_max = value * self._local_max_eng.get()
        logger.debug('%s transmission requested by user,\n pmps max energy adjusted from %s to %s' % (value, self.max_eng.get(), new_max))
        self.max_eng.put(new_max)
        if ( bool(self._apply_cmd.get()) is False) or (self._local_max_eng.get()>4.99) :
            logger.debug('Script terminated by kill command')
            self._user_trans_request.unsubscribe_all()
        
    def apply(self, max_eng=None, duration=None):
        """
        Apply Transmission Override to PMPS.
        This method launches a watcher script to monitor the transmission requested by the user (":BeamParamCntl:ReqBP:Transmission_RBV") and the max energy of the PMPS override is scaled by this transmission to account for the reduced transmission in the beamclass assertation. 
        
        Parameters
        ----------
        max_eng : optional
            writes to 'pmps_override_max_eng' before enabling transmission override. If nothing is specified apply the existing value is applied.
            
        duration : optional
            writes value to 'pmps_override_duration' before enabling transmission override. If nothing is specified apply the existing value is applied.
        """
        if max_eng is not None:
            self._local_max_eng.put(max_eng)
            self.max_eng.put(max_eng)
        if duration is not None:
            self.duration.put(duration)
        self._apply_cmd.put(1)
        time.sleep(1)
        logger.info('PMPS override applied for %s hours. Max energy is %s' % (self.duration.get(), self.max_eng.get()))
        sys.stdout.write("\x1b]2;PMPS\x07")
        if self._facility_mode.get() == 'SC':
            logger.debug('Launching watcher script to adjust transmission to fix beam class.')
            self._user_trans_request.subscribe(self._trans_adjust)
        else:
            logger.debug('Facility mode is not SC, no watcher script launched.')
             
        
    def cancel(self):
        """Cancel Transmission Override in PMPS."""
        self._cancel_cmd.put(1)
        self._local_max_eng.put(5)
        self._user_trans_request.unsubscribe_all()
        
    def status(self):
        """ Get the status of the PMPS control """
        setting_dict={'Override On': self._apply_cmd.get(),
                      'Timing Remaining': self.time_remaining.get(),
                      'TMO Max Energy': self._local_max_eng.get(),
                      'Transmission Set': self._user_trans_request.get(),
                      'PMPS Max Energy': self.max_eng.get()
            }
        _log_positions(setting_dict, Log=False)
    
class TmoBeamLineControls(BaseInterface, GroupDevice):
    """
    This class controls various parameters (rate, transmission) of the K-line.  
    """
    tab_component_names = True
    tab_whitelist = ['apply_beam_parameter', 'beam_rate_watchdog','pmps_watchdog','set_transmission', 'stop_watchdog'] #,'transmission']
    
#    rate = Cpt(EpicsSignalRO, 'TPG:SYS0:1:DST04:RATE_RBV', kind='normal',
#              doc='LCLS BEAM Event Rate [Hz]')
    rate = Cpt(EpicsSignalRO, 'PMPS:KFE:CurrentBP:Rate_RBV', kind='normal',
               doc='LCLS BEAM Event Rate [Hz]')
    photon_eV = Cpt(EpicsSignalRO, 'SIOC:SYS0:ML00:AO628', kind='normal',
                    doc='K-line photon energy')
    mj = Cpt(EpicsSignalRO, 'EM1K0:GMD:HPS:milliJoulesPerPulse', kind='hinted',
             doc='GMD Pulse energy [mJ]')
    mj_x = Cpt(EpicsSignalRO, 'EM2K0:XGMD:HPS:milliJoulesPerPulse', kind='hinted',
             doc='XGMD Pulse energy [mJ]')
    avg_int = Cpt(EpicsSignalRO, 'EM1K0:GMD:HPS:AvgPulseIntensity', kind='hinted',
             doc='GMD Pulse energy [mJ]')
    avg_int_x = Cpt(EpicsSignalRO, 'EM2K0:XGMD:HPS:AvgPulseIntensity', kind='hinted',
             doc='XGMD Pulse energy [mJ]')
    mj_avg = Cpt(EpicsSignalRO, 'EM1K0:GMD:HPS:milliJoulesPerPulse_1MIN_AVG', kind='omitted',
                 doc='GMD Pulse Energy [mJ], 1 min average')
    avg_power = Cpt(EpicsSignalRO, 'EM1K0:GMD:HPS:PWR', kind='hinted',
                    doc='Power measurement from the gmd [mW]')
    avg_power_x = Cpt(EpicsSignalRO, 'EM2K0:XGMD:HPS:PWR', kind='hinted',
                    doc='Power measurement from the gmd [mW]')
    transmission = Cpt(EpicsSignalRO, 'AT1K0:GAS:TRANS_RBV', kind='hinted',
                       doc='AT1K0 Transmission setting from beamline control.')
    facility_mode = Cpt(EpicsSignalRO, 'SIOC:FEES:MP01:FACMODE_RBV', string=True, kind='hinted',
                        doc='Facility Operating Mode [SC or NC]')
    beamclass_current = Cpt(EpicsSignalRO, 'PMPS:KFE:CurrentBP:BeamClass_RBV', kind='hinted',
                            doc='Current Beam Class')
    pmps_override = Cpt(PmpsOverride, 'PMPS:KFE', kind='normal')
    _beamclass_request = Cpt(EpicsSignal, 'PMPS:KFE:BeamParamCntl:ReqBP:BeamClassRanges_RBV', write_pv='PMPS:KFE:BeamParamCntl:ReqBP:BeamClassRanges', kind='omitted',
                            doc='Requested Beam Class')
    _reqBP_apply_cmd = Cpt(EpicsSignal, 'PMPS:KFE:BeamParamCntl:ReqBP:Apply', kind='omitted')
    transmission_scan = Cpt(TmoBeamLinePositioner, 'PMPS:KFE:BeamParamCntl:ReqBP:Transmission', kind='omitted')
    _transmission_cntl = Cpt(EpicsSignal, 'PMPS:KFE:BeamParamCntl:ReqBP:Transmission_RBV', write_pv='PMPS:KFE:BeamParamCntl:ReqBP:Transmission', kind='omitted',
                       doc='AT1K0 Transmission setting from beamline control.')
    
    def elog_beam(self, beam_down: bool, msg=None):
        """
        Post to the elog that the beam is either useable or not usable. The post is tagged with "beam_record." 

        Parameters
        ----------tot_events
        beam_down : bool
            True will post the beam is not useable, False will print that the beam is useable.
        msg : str, optional
            Optional description for the elog post. The default is None.

        """
        if beam_down:
            msg_str = 'Beam is not useable.'
        else:
            msg_str = 'Beam is useable again.'
        if msg is not None:
            msg = msg_str + '\n' + msg
        else:
            msg=msg+str
        elog.post(msg=msg, tags='beam_record')
    
    def apply_beam_parameter(self, transmission=None, beam_class=None):
        """
        Apply the beamline control parameters

        Parameters
        ----------
        transmission : float (between 0 and 1), optional
            Transmission to be applied. If no value is passed, the current value in .transmission will be applied. The default is None.
        beam_class : int, optional
            Beam Class to set. If no value is passed the current value in .beamclass_request is applied. The default is None.

        """
        if transmission is not None:
#           nov 6 2025, we changed the way the override and beam parameters interact, this scaling is not needed now.             
#            if ( (transmission < 1) and self.pmps_override._apply_cmd.get() ):
#                transmission = transmission * (self.pmps_override.max_eng.get()/5)
            self._transmission_cntl.put(transmission)
        if beam_class is None:
            curr_bc = self._beamclass_request.get()
            self._beamclass_request.put(curr_bc)
        else:
            self._beamclass_request.put((1<<beam_class)-1)
            
        self._reqBP_apply_cmd.put(1)
    

    def set_transmission(self, transmission, lazy=False):
        """
        Set the transmission of the gas attenuator. The screen will update with the current transmission unitl a key is pressed. 

        Parameters
        ----------
        transmission : float (between 0 and 1)
            Transmission to be applied.
        lazy : bool, optional
            Option to lazily approach the desired transmission. Adjust the transmission value to keep the beam on, while slowly approaching the desired transmission. 

        """
        def lazy_set(*args, value, timestamp, **kwargs):
            trans_set = self._transmission_cntl.get()
            if value > 1.18 * trans_set:
                self.apply_beam_parameter(transmission=trans_set*1.05)
                return
            elif trans_set > self._intended_trans:
                self.apply_beam_parameter(transmission=0.95*trans_set)
                return
            elif np.isclose(trans_set, self._intended_trans):
                self.transmission.unsubscribe_all()
                return
        
        def display_trans(*args, value, timestamp, **kwargs):
            print(round(value,5), end="\r")
            return
                        
        self.apply_beam_parameter(transmission=transmission, beam_class=None)
        if lazy:
            self._intended_trans = transmission
            self._uid_trans = self.transmission.subscribe(lazy_set)
        uid = self.transmission.subscribe(display_trans)
        input('Current Transmission: (any key to continue) \n')
        self.transmission.unsubscribe(uid)
    
    def _play_wav(self, wav_file, chunk_size=1024):
        
        wf = wave.open(wav_file, 'rb')
        p = pyaudio.PyAudio()
        
        p.terminate()
        
    def beam_rate_watchdog(self, voice='british'):
        """ Monitor the beam rate on the SXR line, and updates to the screen whenever there is a change. """
        def _data_cb(*args, value, timestamp, **kwargs):
            rate_str = pyfiglet.figlet_format(str(round(value)) + " Hz")
            print('Beam Rate to SXR:\n' + rate_str)
            if value == 0:
                print('No Beam on SXR line!!!')
                #self._play_wav(beamDown)
            #else:
            #    self._play_wav(beamUp)
            return
        
        self._uid_rate=self.rate.subscribe(_data_cb)
        sys.stdout.write("\x1b]2;Beam Rate Watcher\x07")
        
    def pmps_watchdog(self, monitor_only=True, diag=False):
        """Monitor the GMD pulse energy and adjust the max_energy value of the PMPS over ride."""
        def _monitor_cb(*args, value, timestamp, **kwargs):
            max_eng = self.pmps_override.max_eng.get()
            if self._pmps_diag:
                print(value, max_eng)#, self.pmps_override.max_eng.get())
            if (value > max_eng):
                if self._pmps_monitor:
                    logger.warning('GMD reading is above PMPS max energy.')
                else:
                    desired_trans = float( '%.2g' % (self.transmission.get()) )
                    logger.warning('GMD reading is above PMPS max energy. Adjusting PMPS Override.')
                    self.pmps_override.apply(max_eng=float('%.2g' % (1.1*max_eng)), duration=(self.pmps_override.time_remaining.get()+0.01))
                    time.sleep(0.2)                
                    self.apply_beam_parameter(transmission=desired_trans)
            return
        def _time_remaining_cb(*args, value, timestamp, **kwargs):
            if np.isclose( value, 1, 2E-4):
                logger.info('PMPS by-pass will expire in 1 hour.')
            if (value < 0.01):
                self.mj_avg.unsubscribe(getattr(self,'_uid_1', None))
                logger.info('Shutdown pmps_watchdog, override has expired.')
                self.pmps_override.time_remaining.unsubscribe(getattr(self, '_uid_2', None))
            return
        
        self._pmps_diag = diag
        self._pmps_monitor = monitor_only
        self._uid_1=self.mj_avg.subscribe(_monitor_cb)
        self._uid_2=self.pmps_override.time_remaining.subscribe(_time_remaining_cb)
        sys.stdout.write("\x1b]2;PMPS WatchDog\x07")
        
        return

    def stop_watchdog(self):
        """ Stop the beam rate and/or pmps watchdog."""
        self.rate.unsubscribe(getattr(self, '_uid_rate', None))
        self.mj_avg.unsubscribe(getattr(self,'_uid_1', None))
        self.pmps_override.time_remaining.unsubscribe(getattr(self, '_uid_2', None))

    def __init__(self, prefix='', name='beam_stats', **kwargs):
        
#        facility_mode = EpicsSignalRO('SIOC:FEES:MP01:FACMODE_RBV', string=True)
#        if facility_mode.get() == 'NC':
#            rate = Cpt(EpicsSignalRO, 'PMPS:KFE:CurrentBP:Rate_RBV', kind='normal',
#                       doc='LCLS BEAM Event Rate [Hz]')
#        else:
#            rate = Cpt(EpicsSignalRO, 'TPG:SYS0:1:DST04:RATE_RBV', kind='normal',
#                       doc='LCLS BEAM Event Rate [Hz]')
        
        super().__init__(prefix=prefix, name=name, **kwargs)
    
#        if self.facility_mode.get() == 'NC':
#            self.rate = Cpt(TmoBeamLinePositioner, 'PMPS:KFE:BeamParamCntl:ReqBP:Rate', kind='normal',
#                            doc='LCLS BEAM Event Rate [Hz]')
    
tmo_beamline_cntl = TmoBeamLineControls(name="tmo_beamline_control")

def set_transmission(transmission):
    """
    Set the transmission of the gas attenuator. 

    Parameters
    ----------
    transmission : float (between 0 and 1)
        Transmission to be applied.

    """
    if transmission>1:
        logger.error('Transmission should be between 0 and 1, not setting transmission.')
        return
    tmo_beamline_cntl.set_transmission(transmission)
    
def set_beam_class(beam_class):
    """
    Set the maximum beam class for the machine. 

    Parameters
    ----------
    beam_class : int (between 0 and 15)
        Max Beam Clas.

    """
    tmo_beamline_cntl.apply_beam_parameter(transmission=None, beam_class=beam_class)

def set_pmps_override(max_energy, duration=12):
    """
    set the beam energy override for the pmps system

    Parameters
    ----------
    max_energy : float
        Max energy to apply for override.
    duration : float, optional
        Duration to apply override. The default is 12.

    Returns
    -------
    None.

    """    
    tmo_beamline_cntl.pmps_override.apply(max_eng=max_energy, duration=duration)
    
class SXRGasAperture(BaseInterface, GroupDevice):
    tab_component_names = True  
    tab_whitelist = ['move','set_aperture', 'position']
    maa_01 = Cpt(GattApertureY, ':01:Y:STATE', kind='hinted')
    maa_02 = Cpt(GattApertureY, ':02:Y:STATE', kind='hinted')
    maa_03 = Cpt(GattApertureY, ':03:Y:STATE', kind='hinted')
    maa_04 = Cpt(GattApertureY, ':04:Y:STATE', kind='hinted')
    
    def move(self, state=None):
        """
        Move aperture motor to the desired state.

        Parameters
        ----------
        state : str or int, optional
            Desired state. The default is None, which will return a list of states

        """
        states = [s.name for s in self.maa_01.states_enum]
        
        if state is None:
            print('valid states are: %s' % states)
        elif (type(state) is int) or (state in states):        
            self.maa_01.move(state)
            self.maa_02.move(state)
            self.maa_03.move(state)
            self.maa_04.move(state)
        else:
            logger.error('%s is not a valid state, valid states are: %s' % (state, states))
        
    def set_aperture(self, state):
        """ Set the aperture array."""
        self.move(state)
        
    @property
    def position(self):
        return (self.maa_01.position, self.maa_02.position, self.maa_03.position, self.maa_04.position)
    
    @property
    def moving(self):
        return any([self.maa_01.moving, self.maa_02.moving, self.maa_03.moving, self.maa_04.moving])
    
class SXRGasAtt(BaseInterface, GroupDevice):
    tab_component_names = True
    tab_whitelist = ['setup_mode']

    transmission = Cpt(EpicsSignal, ':TRANS_RBV', write_pv=':TRANS_SP', kind='hinted', 
                       doc='')
    arb_req = Cpt(EpicsSignalRO, ':TRANS_REQ_RBV', kind='hinted', 
                  doc='')
    pressure = Cpt(EpicsSignal, ':GCM:82:PRESS_RBV', write_pv=':CNTRL:SP', kind='hinted',
                   doc='')
    pressure_setpoint_rbv = Cpt(EpicsSignalRO, ':CNTRL:SP_RBV', kind='omitted', 
                            doc='')
    mode = Cpt(EpicsSignal, ':MODE_RBV', write_pv=':MODE', string=True, kind='hinted', 
               doc='')
    control_enable = Cpt(EpicsSignal, ':CNTRL:ON_RBV', write_pv=':CNTRL:ON', kind='hinted', 
                         doc='')  
    pressure_control_enable = Cpt(EpicsSignal, ':MODE:PressureControl_RBV', write_pv=':MODE:PressureControl', kind='hinted', 
                                  doc='')
    gas_type = Cpt(EpicsSignalRO, ':GAS_TYPE_RBV', string=True, kind='hinted', 
                   doc='')
    at_target = Cpt(EpicsSignalRO, ':AtTarget_RBV', string=True, kind='hinted', 
                    doc='')
    moving = Cpt(EpicsSignalRO, ':Moving_RBV', string=True, kind='hinted', 
                 doc='')
    gas_att_ok = Cpt(EpicsSignalRO, ':OK_RBV', string=True, kind='hinted', 
                     doc='')  
    transmission_setpoint_rbv = Cpt(EpicsSignalRO, ':TRANS_SP_RBV', kind='omitted', 
                                         doc='')
    pressure_control_valve = Cpt(EpicsSignalRO, ':VCN:70:POS_REQ_RBV', kind='omitted', 
                                 doc='')
    valve_n2 = Cpt(VVC, ':VVC:72', kind='hinted', doc='')
    valve_ar = Cpt(VVC, ':VVC:71', kind='hinted', doc='')
    valve_pressure_control = Cpt(VCN, ':VCN:70', kind='omitted', doc='')
    
    apperture_array = Cpt(SXRGasAperture, '_MAA', kind='hinted', doc='aperture array for gas attenuator.')
    
    def setup_mode(self, mode, control_type='transmission', gas_type=None, set_point=None):
        """
        Setup AT1K0 gas attenuator to work in "PMPS" or "Local" mode, with either "transmission control" or "pressure control"

        Parameters
        ----------
        mode : str, either "PMPS" or "Local"
            Mode for attenuator.
        control_type : str, optional
            Set control type in "Local" mode, either "transmission" or "pressure" control. The default is 'transmission'.
        gas_type : str, optional
            Change gas type to "N2" or "Ar". The default is None. If None is passed the attenuator uses the current gas.

        """
        if ( mode.lower() not in ['pmps','local'] ):
            print('unrecognizied mode, options are "PMPS" or "Local"')
            return
        elif mode.lower() == "local":
            if control_type.lower() not in ['transmission', 'pressure']:
                print('unrecognizied control type, options are "transmission" or "pressure"')
                return
        if (gas_type not in ['N2', 'Ar', None]):
            print('unrecognizied gas type, options are "N2", "Ar", or None')
            return
        
        if gas_type is not None:
            self.transmission.put(1)
            self.valve_ar.open_command.put(0)
            self.valve_n2.open_command.put(0)
            time.sleep(1)
            if gas_type == 'N2':
                  self.valve_n2.open_command.put(1)
            elif gas_type == 'Ar':
                  self.valve_ar.open_command.put(1)
            time.sleep(1)
        
        if mode.lower() == 'pmps':
            self.pressure_control_enable.put(0)
            self.control_enable.put(0)
            self.mode.put('PMPS')      
        
        elif mode.lower() == 'local':
            if not self.valve_pressure_control.interlock_ok.get():
                logger.error('Mode not set. '+self.valve_pressure_control.name+'('+self.valve_pressure_control.prefix+') interlock has an issue. Check vacuum system.')
                return
            self.mode.put('Local')
            if control_type.lower() == 'transmission':
                self.pressure_control_enable.put(0)
                self.control_enable.put(1)
                if set_point is not None:
                    self.transmission.put(set_point)
            elif control_type.lower() == 'pressure':
                self.pressure_control_enable.put(1)
                self.control_enable.put(1)
                self.pressure.put(0)
                if set_point is not None:
                    self.pressure.put(set_point)
                 

at1k0 = SXRGasAtt('AT1K0:GAS', name='at1k0')
    
class UndSeg(GroupDevice):
    """
    Undulator Segement Device
    
    """
    k_act = UCpt(EpicsSignalRO)
    k_des = UCpt(EpicsSignalRO)
    x_bpm = UCpt(EpicsSignalRO)
    y_bpm = UCpt(EpicsSignalRO)
    x_off = UCpt(EpicsSignalRO)
    y_off = UCpt(EpicsSignalRO)

    def __init__(self, prefix, **kwargs):
        UCpt.collect_prefixes(self, kwargs)
        super().__init__(prefix, **kwargs)

def _build_und_list(segs):
    defn = OrderedDict()
    fac_mode = tmo_beamline_cntl.facility_mode.get()
    fac_mode = fac_mode if fac_mode=='SC' else 'CU'
    for und_ind in segs:
        k_act_pv = 'USEG:UNDS:%d50:KAct' % (und_ind)
        k_des_pv = 'USEG:UNDS:%d50:KDes' % (und_ind)
        x_bpm_pv = 'BPMS:UNDS:%d90:X%sS1H' % (und_ind,fac_mode)
        y_bpm_pv = 'BPMS:UNDS:%d90:Y%sS1H' % (und_ind,fac_mode)
        x_off_d = 'BPMS:UNDS:%d90:XOFF.D' % (und_ind)
        y_off_d = 'BPMS:UNDS:%d90:YOFF.D' % (und_ind)
        name = 'u' + str(und_ind)
        defn[name.lower()] = Cpt(UndSeg, "", k_act_prefix=k_act_pv, k_des_prefix=k_des_pv,
                                 x_bpm_prefix=x_bpm_pv, y_bpm_prefix=y_bpm_pv,
                                 x_off_prefix=x_off_d, y_off_prefix=y_off_d,
                                 kind='hinted')
    return defn
       
class UndulatorParameter(BaseInterface, GroupDevice):
    """
    Properties of Undulator Line
        k_act - K value of each undulator
        x_bpm - value from x-BPM 
        y_bpm - value from y-BPM
        x_off - x offset ('BPMS:UNDS:%%90:XOFF.D')
        y_off - y offset ('BPMS:UNDS:%%90:YOFF.D')
    """
    tab_component_names = True
    tab_whitelist = ['get_k','get_bpm', 'get_offset']
    
    def get_k(self, log=False, plot=False, msg=None, **kwargs):
        """
        Get current position of the undulator K.
        Parameters:
        -------------
        log - boolean to log the positions in the elog 
        msg - optional argument to append a message to the elog with the KB positions 
        plot - display a plot of the positions
        """
        curr_positions = OrderedDict()
        targ_positions = OrderedDict()
        segs = []
        for keys in dir(self):
            und = getattr(self, keys)
            if hasattr(und, 'k_act'):
                curr_positions[keys] = und.k_act.get()
                targ_positions[keys] = und.k_des.get()
                segs.append(int(keys[1:]))
        kwargs['tags'] = kwargs.get('tags', ' ') + ' undulator_k'                  
        
        if plot:
            fig = plt.figure()
            plt.plot(segs, list(curr_positions.values()))
            plt.ylabel('K-value')
            plt.xlabel('Undulator Segment')
            plt.show()
            
            fig.tight_layout()
            fig.savefig('/cds/home/opr/tmoopr/tmoopr_elog_plots/unds.png')
            kwargs['attachments'] = ['/cds/home/opr/tmoopr/tmoopr_elog_plots/unds.png']
        
        _log_positions(curr_positions, log, msg, set_points=targ_positions, **kwargs) 

    def get_bpm(self, save_ref=False):
        """
        Get current beam position (x,y) from the bpm on each undulator section.
        """
        def _data_cb(*args, value, timestamp, obj, **kwargs):
            xy = obj.name[-5]
            und_num = obj.name[-8:-6]
            positions[xy][und_num] = value
            #ax.plot(u_segs, list(positions[xy].values()))
            if plt.fignum_exists(888):
                lines[xy].set_ydata(list(positions[xy].values()))
                #fig.canvas.draw()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            else:
                for key in dir(self):
                    und = getattr(self, key)
                    if hasattr(und, 'x_bpm'):
                        und.x_bpm.unsubscribe_all()
                    if hasattr(und, 'y_bpm'):
                        und.y_bpm.unsubscribe_all()
          
        positions = {}
        positions['x'] = {}
        positions['y'] = {}
        lines = {}
        segs=[]

        for key in dir(self):
            und = getattr(self, key)
            if hasattr(und, 'x_bpm'):
                segs.append(int(key[1:]))
                #positions['x'][und.name[1:3]]=und.x_bpm.get()
                positions['x'][key[1:]]=und.x_bpm.get()
            if hasattr(und, 'y_bpm'):
                #positions['y'][und.name[1:3]]=und.y_bpm.get()
                positions['y'][key[1:]]=und.y_bpm.get()
        
        if save_ref:
            with h5py.File('/cds/home/opr/tmoopr/tmoopr_elog_plots/bpm_ref.h5','w') as hf_dat:
                hf_dat['x']=list(positions['x'].values())
                hf_dat['y']=list(positions['y'].values())
                hf_dat['timestamp']=[datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")]
        
        plt.ion()
        fig = plt.figure(888)
        ax = fig.add_subplot(111)
        
        with h5py.File('/cds/home/opr/tmoopr/tmoopr_elog_plots/bpm_ref.h5','r') as hf_dat:
            ax.plot(segs, hf_dat['x'][:], label=('x %s'%hf_dat['timestamp'][0]), marker='o', linestyle='None')
            ax.plot(segs, hf_dat['y'][:], label=('y %s'%hf_dat['timestamp'][0]), marker='o', linestyle='None')
        
        lx, = ax.plot(segs, list(positions['x'].values()),'C0',label='x')
        ly, = ax.plot(segs, list(positions['y'].values()),'C1',label='y')
        lines['x'] = lx
        lines['y'] = ly
        plt.ylabel('BPM')
        plt.xlabel('Undulator Segment')
        plt.legend()
#        plt.legend(['x', 'y'])
        plt.draw()
                
        uid={}
        for key in dir(self):
            und = getattr(self, key)
            if hasattr(und, 'x_bpm'):
                uid[und.name+'x'] = und.x_bpm.subscribe(_data_cb)
            if hasattr(und, 'y_bpm'):
                uid[und.name+'y'] = und.y_bpm.subscribe(_data_cb)
    
    def get_offset(self, log=False, msg=None, **kwargs):
        """
        Plot X_OFF and Y_OFF
        
        """
        x_positions = OrderedDict()
        y_positions = OrderedDict()
        segs = []
        for keys in dir(self):
            und = getattr(self, keys)
            if hasattr(und, 'x_off'):
                x_positions[keys] = und.x_off.get()
                y_positions[keys] = und.y_off.get()
                segs.append(int(keys[1:]))
        
        fig=plt.figure()
        plt.plot(segs, list(x_positions.values()))
        plt.plot(segs, list(y_positions.values()))
        plt.ylabel('Offset')
        plt.xlabel('Undulator Segment')
        plt.legend(['x', 'y'])
        plt.show()
        if log or ( not(msg is None) ):
            curr_positions = {'u34_x':x_positions['u34'], 'u34_y':y_positions['u34'],
                              'u47_x':x_positions['u47'], 'u47_y':y_positions['u47']}
            fig.tight_layout()
            fig.savefig('/cds/home/opr/tmoopr/tmoopr_elog_plots/unds.png')
            plt.close(fig)
            kwargs['attachments'] = ['/cds/home/opr/tmoopr/tmoopr_elog_plots/unds.png']
            kwargs['tags'] = kwargs.get('tags', ' ') + ' undulator_offsets'  
            _log_positions(curr_positions, log, msg, **kwargs)
            
        

u_segs = list(range(20,35)) + list(range(36,48))
Unds = type('Unds',(UndulatorParameter, BaseInterface, Device), _build_und_list(u_segs))
unds = Unds(name='undulators')
#unds = UndulatorParameter()

def st1k0_follow_st3k4():
    def transition_check(*, old_value, value, **kwargs):
        logger.debug('old_value: %s, value: %s' % (old_value, value))
        if ((value != old_value) and (value>0) ): #value changed and stopper is closing
           # st1k0.state('In')
           logger.debug('insert st1k0')
    st3k4_pps.in_signal.subscribe(transition_check)
    sys.stdout.write("\x1b]2;ST3K4 Follow\x07")
        

def get_stprs():
    state=['OUT','IN', 'Moving', 'Invalid']
    curr_positions = {
        st1k0.name: st1k0.state.position,
        st3k4_pps.name: state[(1-st3k4_pps.out_signal.get())],
        st4k4.name:state[st4k4.position_state.get()]
        }
    _log_positions(curr_positions, log=False, msg=None)
    
def stop_xray():
    """
    This function puts st1k0 in to block the beam. 
    """
    beam_class = tmo_beamline_cntl.beamclass_current.get()
    tmo_beamline_cntl.apply_beam_parameter(beam_class=1)
    st1k0.state.mv('IN', wait=True)
    tmo_beamline_cntl.apply_beam_parameter(beam_class=beam_class)
    
def get_rtds():
    """
    Get the current state of the rtdsk0 filters
    """
    curr_positions= {
        rtdsk0.mpa1.name:rtdsk0.mpa1.position,
        rtdsk0.mpa2.name:rtdsk0.mpa2.position}    
    _log_positions(curr_positions, log=False, msg=None)

def set_rtds(state=None):
    """
    Function to change the RTDSK0 filter positions
    Parameters:
    ---------------
    state - 
    if no state is provided, the fucntion returns the current state of the rtds filters.
    if a single number is passed it is treated as a bit mask, where the first bit is filter 1 state and the second bit is filter 2).
    if a binary list is passed [#,#], the entries provide the state for each filter (0=out, 1=in)
    """
    if state == None:
        get_rtds()
    elif (np.size(state) == 1):
        if (state == 0):
            rtdsk0.mpa1.remove()
            rtdsk0.mpa2.remove()
        elif (state == 1):
            rtdsk0.mpa1.insert()
            rtdsk0.mpa2.remove()
        elif (state == 2):
            rtdsk0.mpa1.remove()
            rtdsk0.mpa2.insert()
        elif (state == 3):
            rtdsk0.mpa1.insert()
            rtdsk0.mpa2.insert()
    elif (np.size(state) == 2):
        if state[0]:
            rtdsk0.mpa1.insert()
        else:
            rtdsk0.mpa1.remove()
        if state[1]:
            rtdsk0.mpa2.insert()
        else:
            rtdsk0.mpa2.remove()

def get_vac_valves():
    """
    Get the status of the vacuum valves along the TMO beamline.
    """
    tmo_hvac.valve_status()

def get_unds(log=False, msg=None):
    """
    Get current position of the undulator K.
    Parameters:
    -------------
    log - boolean to log the positions in the elog 
    msg - optional argument to append a message to the elog with the KB positions 
    
    """
    unds.get_k(log=log, msg=msg, plot=False)
 
class GMDPreAmp(BaseInterface, Device):
    tab_component_names = True
    ion_pre_att = Cpt(EpicsSignal, ':AMP_PREATTN1_RBV', write_pv=':AMP_PREATTN1', string=True, kind='hinted')
    ion_post_att = Cpt(EpicsSignal, ':AMP_POSATTN1_RBV', write_pv=':AMP_POSATTN1', string=True, kind='hinted')
    elec1_pre_att = Cpt(EpicsSignal, ':AMP_PREATTN2_RBV', write_pv=':AMP_PREATTN2', string=True, kind='hinted')
    elec1_post_att = Cpt(EpicsSignal, ':AMP_POSATTN2_RBV', write_pv=':AMP_POSATTN2', string=True, kind='hinted')
    elec2_pre_att = Cpt(EpicsSignal, ':AMP_PREATTN3_RBV', write_pv=':AMP_PREATTN3', string=True, kind='hinted')
    elec2_post_att = Cpt(EpicsSignal, ':AMP_POSATTN3_RBV', write_pv=':AMP_POSATTN3', string=True, kind='hinted')
    spare_pre_att = Cpt(EpicsSignal, ':AMP_PREATTN4_RBV', write_pv=':AMP_PREATTN4', string=True, kind='hinted')
    spare_post_att = Cpt(EpicsSignal, ':AMP_POSATTN4_RBV', write_pv=':AMP_POSATTN4', string=True, kind='hinted')
    
class Keithley(BaseInterface, Device):
    tab_component_names = True 
    
    avg_enable = Cpt(EpicsSignal, ':AvgEnable', kind='hinted', doc='')
    avg_count = Cpt(EpicsSignal, ':GetAvgCount', write_pv='PutAvgCount', kind='hinted', doc='')
    avg_mode = Cpt(EpicsSignal, ':PutAvgType', string=True, kind='hinted', doc='')
    auto_range = Cpt(EpicsSignal, ':PutAutoRange', kind='hinted', doc='')
    current_range = Cpt(EpicsSignal, ':SelectCurrentRange', string=True, kind='hinted', doc='')
    damping = Cpt(EpicsSignal, ':PutDamping', string=True, kind='hinted', doc='')
    integration_time = Cpt(EpicsSignal, ':GetIntCycles', write_pv=':PutIntCycles', kind='hinted', doc='')
    measurement_function = Cpt(EpicsSignal, ':PutFunction', string=True, kind='hinted', doc='')
    reading_rate = Cpt(EpicsSignal, ':Reading.SCAN', string=True, kind='hinted', doc='')
    reading = Cpt(EpicsSignalRO, ':Reading', kind='hinted', doc='')
    zero_correct = Cpt(EpicsSignal, ':ZeroCorrect', kind='hinted', doc='')

class GMDPulseIntegration(BaseInterface, Device):
    tab_component_names = True 
    
    offset_enu = Cpt(EpicsSignal, ':OffsetEnURBV', write_pv=':OffsetEnU', kind='hinted', doc='')
    offset_env = Cpt(EpicsSignal, ':OffsetEnVRBV', write_pv=':OffsetEnV', kind='hinted', doc='')
    scale_en = Cpt(EpicsSignal, ':CoefEnRBV', write_pv=':CoefEn', kind='hinted', doc='')
    status = Cpt(EpicsSignalRO, ':Status', kind='hinted', doc='')
    auto_calibrate_activation = Cpt(EpicsSignal, ':PulseAutoScale', kind='omitted', doc='')
    calibrate_seed_factor = Cpt(EpicsSignal, ':PulseSeedFactor', kind='omitted',doc='')
    calibrate_max_energy = Cpt(EpicsSignal, ':MaxEn', kind='omitted', doc='')
    calibrate_energy_thresh = Cpt(EpicsSignal, ':EnThresholdRBV', write_pv=':EnThreshold', kind='omitted', doc='')
    
    def enable_auto_calibrate(self, seed_factor=None, max_energy=None, threshold=None):
        if seed_factor is not None:
            self.calibrate_seed_factor.put(seed_factor)
        if max_energy is not None:
            self.calibrate_max_energy.put(max_energy)
        if threshold is not None:
            self.calibrate_energy_thresh.put(threshold)
        self.auto_calibrate_activation.put(1)
                    
    def disable_auto_calibrate(self):
        self.auto_calibrate_activation.put(0)
    
    
class GMD(BaseInterface, Device):
    tab_component_names = True
    avg_int = Cpt(EpicsSignalRO, ':HPS:AvgPulseIntensity', kind='hinted',
             doc='Avg Pulse energy [mJ]')
    mj = Cpt(EpicsSignalRO, ':HPS:milliJoulesPerPulse', kind='hinted',
             doc='Pulse energy [mJ]')
    photons = Cpt(EpicsSignalRO, ':HPS:AvgPhotonsPerPulse', kind='hinted',
             doc='photons')
    transmission = Cpt(EpicsSignalRO, ':HPS:AvgTransmission', kind='hinted',
             doc='transmission')
    avg_power = Cpt(EpicsSignalRO, ':HPS:PWR', kind='hinted',
                    doc='average power measurment in mW.')
    gas_type = Cpt(EpicsSignalRO, ':GAS_TYPE_RBV', string=True, kind='hinted',
                   doc='Gas Type')
    mean_charge = Cpt(EpicsSignal, ':HPS:MeanCharge', write_pv=':HPS:MeanCharge:Manual', kind='normal',
                      doc='Mean Charge used in energy calculation')
    xsection = Cpt(EpicsSignal, ':HPS:CrossSection', write_pv=':HPS:CrossSection:Manual', kind='normal',
                   doc='Photoionization cross section used in energy calculation')
    keithley_sum = Cpt(EpicsSignalRO, ':HPS:KeithleySum', kind='normal',
                       doc='')
    pressure = Cpt(EpicsSignalRO, ':GSR:1:Calib:Pressure:Calc', kind='normal',
                   doc='Gas pressure in energy monitor')
    mean_charge_source = Cpt(EpicsSignal, ':HPS:MeanCharge:Source', string=True, kind='omitted',
                             doc='Source value of mean charge (Gas Table or Manual) for energy calculation')
    xsection_source = Cpt(EpicsSignal, ':HPS:MeanCharge:Source', string=True, kind='omitted',
                          doc='Source value of photoionization cross section (Gas Table or Manual) for energy calculation')
    temperature = Cpt(EpicsSignalRO, ':RTD:1:TEMP_RBV', kind='hinted', doc='')
    beam_position_x = Cpt(EpicsSignalRO, ':HPS:PosXSLOW', kind='hinted',
                          doc='beam position x in GMD')
    beam_position_y = Cpt(EpicsSignalRO, ':HPS:PosYSLOW', kind='hinted',
                          doc='beam position y in GMD')
    preamp = Cpt(GMDPreAmp, ':HPS', kind='omitted')
    keithley1 = Cpt(Keithley, ':ETM:01', kind='omitted', doc='')
    keithley2 = Cpt(Keithley, ':ETM:02', kind='omitted', doc='')
    pulse_int = Cpt(GMDPulseIntegration, ':HPS', kind='omitted', doc='')
    
    def __init__(self, *args, **kwargs):
        self.flyer_attr=['mj','avg_int'] #attributes for flying
        self._start_time=None #indicator for start of flying
        self._acquiring = False #indicator for flyer acquiring
        self._paused = False #indicator for pausing flyer
        return super().__init__(*args, **kwargs)
           
    def _data_cb(self, value, timestamp, **kwargs):
        obj = kwargs['obj']
        self.data[obj.name]['value'].append(value)
        self.data[obj.name]['timestamp'].append(timestamp)
    
    def stage(self):
        self.data=OrderedDict()
        self.data[self.avg_int.name]={'value':[], 'timestamp':[]}
        self.data[self.mj.name]={'value':[], 'timestamp':[]}
        self.uid1=self.avg_int.subscribe(self._data_cb)
        self.uid2=self.mj.subscribe(self._data_cb)        
        return[self]
    
    def describe(self):
        data = {
            self.avg_int.name: {'source':self.avg_int.pvname, 'dtype':'array','shape':[], 'units':'mj'},
            self.mj.name: {'source':self.mj.pvname, 'dtype':'array','shape':[], 'units':'mj'},
#            self.avg_int.name: {'source':self.avg_int.pvname, 'dtype':'string','shape':[1,10], 'units':'mj'},
#            self.mj.name: {'source':self.mj.pvname, 'dtype':'string','shape':[1,10], 'units':'mj'},
            }
        return data
    
    def trigger(self):
        self.data[self.avg_int.name]={'value':[], 'timestamp':[]}
        self.data[self.mj.name]={'value':[], 'timestamp':[]} 
        return DeviceStatus(self, done=True, success=True)
    
    def read(self):
        return self.data
    
    def unstage(self):
        self.avg_int.unsubscribe(self.uid1)
        self.mj.unsubscribe(self.uid2)
        return [self]
    
    def kickoff(self):
        logger.debug('called .kickoff')
        #if self._completion_status is not None and not self._completion_status.done:
        if self._acquiring:
            raise RuntimeError("Kicking off a second time?!")      
        
        self.data={}
        for attr in self.flyer_attr:
            obj = getattr(self, attr)
            self.data[obj.name]={'value':[], 'timestamp':[]}
        
#        st = DeviceStatus(device=self)
#        self._completion_status = st
#        st.set_finished()
        
        for attr in self.flyer_attr:
            obj = getattr(self, attr)
            obj.subscribe(self._data_cb)
            logger.debug('subscribe %s' % obj.name)
        
        self._start_time = time.time()
        self._acquiring = True
        self._paused = False
        
        kickoff_st = DeviceStatus(device=self)
        kickoff_st.set_finished()
        return kickoff_st
    
    def pause(self):
        '''Pause acquisition'''
        logger.debug('called .pause')
        if not self._acquiring:
            # nothing to do
            return
        
        for attr in self.flyer_attr:
            obj = getattr(self, attr)
            obj.unsubscribe_all()
            logger.debug('unsubscribe %s' % obj.name)
        self._paused = True
    
    def resume(self):
        '''Resume acquisition'''
        logger.debug('called .resume')
        if not self._acquiring:
            # nothing to do
            return
        if not self._paused:
            #do nothings
            return
        #restart subscription
        for attr in self.flyer_attr:
            obj = getattr(self, attr)
            obj.subscribe(self._data_cb)
            logger.debug('subscribe %s' % obj.name)
        self._paused = False
        
    def complete(self):
        logger.debug('called .complete')
        #if self._completion_status is None:
        if not self._acquiring:
            raise RuntimeError("No collection in progress")
        
        for attr in self.flyer_attr:
            obj = getattr(self, attr)
            obj.unsubscribe_all()
            logger.debug('unsubscribe %s' % obj.name)
        
        self._acquiring = False #not acquiring
        self._paused = False #not paused
        
        completion_status = DeviceStatus(self)
        completion_status.set_finished()
        return completion_status

#        if not self._completion_status.done:
#            logger.debug('set completion to done.')
#            self._completion_status.set_finished()
#        return self._completion_status
            
    def describe_collect(self):
        logger.debug('called .describe_collect')
        data={}
        for attr in self.flyer_attr:
            obj = getattr(self, attr)
            data[obj.name]={'source':obj.pvname, 'dtype':'number','shape':[], 'units':'mj'}
        return {self.name: data}
    
    def collect(self):
        logger.debug('called .collect')
        if self._acquiring:
            raise RuntimeError('Acquisition still in progress. Call complete() first.')
        timestamps={}
        value={}
        for attr, data in self.data.items():
            timestamps[attr]=data['timestamp']
            value[attr]=data['value']
        yield dict(time=self._start_time,
            timestamps=timestamps,
            data=value)
        
gmd = GMD('EM1K0:GMD', name='gmd')
xgmd = GMD('EM2K0:XGMD', name='xgmd')

class TwoPulseTwoColor(BaseInterface, Device):
    tab_component_names=True
    
    p1 = Cpt(EpicsSignalRO, ':P1:AvgPulseIntensity', kind='normal',
             doc='pulse energy for longer wavelength pulse')
    p2 = Cpt(EpicsSignalRO, ':P2:AvgPulseIntensity', kind='normal',
             doc='pulse energy for shorter wavelength pulse')
    K0 = Cpt(EpicsSignalRO, ':K0', kind='normal',
             doc='ask stefan')
    K1 = Cpt(EpicsSignalRO, ':K1', kind='normal',
             doc='ask stefan')
    K2 = Cpt(EpicsSignalRO, ':K2', kind='normal',
             doc='ask stefan')
    ratio = Cpt(EpicsSignal,':RATIO', kind='normal',
                doc='ratio betwee pulse 1 and pulse 2 photon energies.')
    
    def __init__(self, *args, **kwargs):
        self.flyer_attr=['p1','p2']
        self._start_time=None #indicator for start of flying
        self._acquiring = False #indicator for flyer acquiring
        self._paused = False #indicator for pausing flyer
        return super().__init__(*args, **kwargs)
    
    def _data_cb(self, value, timestamp, **kwargs):
        obj = kwargs["obj"]
        self.data[obj.name]['value'].append(value)
        self.data[obj.name]['timestamp'].append(timestamp)
    
    def stage(self):
        self.data=OrderedDict()
        self.data[self.p1.name]={'value':[], 'timestamp':[]}
        self.data[self.p2.name]={'value':[], 'timestamp':[]}
        self.uid1=self.p1.subscribe(self._data_cb)
        self.uid2=self.p2.subscribe(self._data_cb)        
        return[self]
    
    def trigger(self):
        self.data[self.p1.name]={'value':[], 'timestamp':[]}
        self.data[self.p2.name]={'value':[], 'timestamp':[]} 
        return DeviceStatus(self, done=True, success=True)
    
    def read(self):
        return self.data
    
    def unstage(self):
        self.p1.unsubscribe(self.uid1)
        self.p2.unsubscribe(self.uid2)
        return [self]
    
    def describe(self):
        data = {
            self.p1.name: {'source':self.p1.pvname, 'dtype':'array','shape':[1,10], 'units':'mj'},
            self.p2.name: {'source':self.p2.pvname, 'dtype':'array','shape':[1,10], 'units':'mj'},
            }
        return data
    
    def kickoff(self):
        logger.debug('called %s.kickoff' % self.name)
        #if self._completion_status is not None and not self._completion_status.done:
        if self._acquiring:
            raise RuntimeError("Kicking off a second time?!")      
        
        self.data={}
        for attr in self.flyer_attr:
            obj = getattr(self, attr)
            self.data[obj.name]={'value':[], 'timestamp':[]}
        
        for attr in self.flyer_attr:
            obj = getattr(self, attr)
            obj.subscribe(self._data_cb)
            logger.debug('subscribe %s' % obj.name)
        
        self._start_time = time.time()
        self._acquiring = True
        self._paused = False
        
        kickoff_st = DeviceStatus(device=self)
        kickoff_st.set_finished()
        return kickoff_st
    
    def pause(self):
        '''Pause acquisition'''
        logger.debug('called .pause')
        if not self._acquiring:
            # nothing to do
            return
        
        for attr in self.flyer_attr:
            obj = getattr(self, attr)
            obj.unsubscribe_all()
            logger.debug('unsubscribe %s' % obj.name)
        self._paused = True
    
    def resume(self):
        '''Resume acquisition'''
        logger.debug('called .resume')
        if not self._acquiring:
            # nothing to do
            return
        if not self._paused:
            #do nothings
            return
        #restart subscription
        for attr in self.flyer_attr:
            obj = getattr(self, attr)
            obj.subscribe(self._data_cb)
            logger.debug('subscribe %s' % obj.name)
        self._paused = False
    
    def complete(self):
        logger.debug('called .complete')
        #if self._completion_status is None:
        if not self._acquiring:
            raise RuntimeError("No collection in progress")
        
        for attr in self.flyer_attr:
            obj = getattr(self, attr)
            obj.unsubscribe_all()
            logger.debug('unsubscribe %s' % obj.name)
        
        self._acquiring = False #not acquiring
        self._paused = False #not paused
        
        completion_status = DeviceStatus(self)
        completion_status.set_finished()
        return completion_status
            
    def describe_collect(self):
        logger.debug('called .describe_collect')
        data={}
        for attr in self.flyer_attr:
            obj = getattr(self, attr)
            data[obj.name]={'source':obj.pvname, 'dtype':'number','shape':[], 'units':'mj'}
        return {self.name: data}
    
    def collect(self):
        logger.debug('called .collect')
        if self._acquiring:
            raise RuntimeError('Acquisition still in progress. Call complete() first.')
        timestamps={}
        value={}
        for attr, data in self.data.items():
            timestamps[attr]=data['timestamp']
            value[attr]=data['value']
        yield dict(time=self._start_time,
            timestamps=timestamps,
            data=value)
    
two_pulse_two_color = TwoPulseTwoColor('TPC:KFE', name='two_pulse_two_color')
