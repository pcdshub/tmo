from ophyd.device import Component as Cpt
from ophyd.device import FormattedComponent as FCpt
from pcdsdevices.device import UnrelatedComponent as UCpt
from pcdsdevices.device import GroupDevice, Device
from pcdsdevices.attenuator import GattApertureY
from pcdsdevices.interface import BaseInterface
from ophyd.signal import EpicsSignal, EpicsSignalRO
from pcdsdevices.pv_positioner import PVPositioner, PVPositionerIsClose
from pcdsdevices.valve import VVC, VCN
from epics import caget, caput, cainfo
from tmo.db import tmo_cvmi, tmo_lamp, tmo_mrco, tv1k4_vgc_01, tv1k4_vfs_01, tv1k4_vgc_02, st3k4_pps_vgc_01, mr2k4_kbo_vwc_01, mr3k4_kbo_vwc_01, mr3k4_kbo_vfs_01, li3k4_ip1_vwc_01, pc5k4_vwc_01, rtdsk0
from tmo.db import im2k0, st1k4
from tmo.tmo_hvac import tmo_hvac
from pcdsdevices.mpod_apalis import MPODApalisChannel
from itertools import chain
import matplotlib.pyplot as plt
import pyfiglet

from elog import HutchELog as elog_temp
elog = elog_temp.from_registry()
import pandas as pd
import numpy as np
from collections import OrderedDict
import time
import h5py
import logging
logger = logging.getLogger(__name__)

def _log_positions(curr_positions,log, msg=None, **kwargs):
    positionDF = pd.DataFrame({"Device:": curr_positions.keys(), "Position:": curr_positions.values()})
    print (positionDF)
    
    if log:
        if (msg is not None):
            msg = msg + "\n" + positionDF.to_string()
        else:
            msg = positionDF.to_string()
        elog.post(msg, **kwargs)
 
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
    
        
gmd = GMD('EM1K0:GMD', name='gmd')
xgmd = GMD('EM2K0:XGMD', name='xgmd')
   
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
    """
    tab_component_names = True
    tab_whitelist = ['apply','cancel']
    
    max_eng = Cpt(EpicsSignal, ':IntensityJF_RBV', write_pv=':IntensityJF', kind = 'hinted',
                                doc='PMPS Override Max Credible Energy [mJ]')
    duration =  Cpt(EpicsSignal, ':DurationJF_RBV', write_pv=':DurationJF', kind = 'hinted',
                                  doc='PMPS Override Timeout Duration [Hrs]')
    time_remaining = Cpt(EpicsSignalRO, ':TimeRemainJF_RBV', kind='hinted',
                                       doc='PMPS Override Time Remaining [Hrs]')
    _apply_cmd = Cpt(EpicsSignal, ':ApplyJF_RBV', write_pv=':ApplyJF', kind='omitted')
    _cancel_cmd = Cpt(EpicsSignal, ':CancelJF', kind='omitted')
    
    def apply(self, max_eng=None, duration=None):
        """
        Apply Transmission Override to PMPS.
        
        Parameters
        ----------
        max_eng : optional
            writes to 'pmps_override_max_eng' before enabling transmission override. If nothing is specified apply the existing value is applied.
            
        duration : optional
            writes value to 'pmps_override_duration' before enabling transmission override. If nothing is specified apply the existing value is applied.
        """
        if max_eng is not None:
            self.max_eng.put(max_eng)
        if duration is not None:
            self.duration.put(duration)
        self._apply_cmd.put(1)
        
    def cancel(self):
        """Cancel Transmission Override in PMPS."""
        self._cancel_cmd.put(1)
    
class TmoBeamLineControls(BaseInterface, GroupDevice):
    """
    This class controls various parameters (rate, transmission) of the K-line.  
    """
    tab_component_names = True
    tab_whitelist = ['apply_beam_parameter', 'beam_rate_watcher'] #,'transmission']
    
    rate = Cpt(EpicsSignalRO, 'TPG:SYS0:1:DST04:RATE_RBV', kind='normal',
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
    transmission = Cpt(EpicsSignalRO, 'AT1K0:GAS:TRANS_RBV', kind='hinted',
                       doc='AT1K0 Transmission setting from beamline control.')
    facility_mode = Cpt(EpicsSignalRO, 'SIOC:FEES:MP01:FACMODE_RBV', string=True, kind='hinted',
                        doc='Facility Operating Mode [SC or NC]')
    beamclass_current = Cpt(EpicsSignalRO, 'PMPS:KFE:CurrentBP:BeamClass_RBV', kind='hinted',
                            doc='Current Beam Class')
    pmps_override = Cpt(PmpsOverride, 'PMPS:KFE', kind='normal')
    _beamclass_request = Cpt(EpicsSignalRO, 'PMPS:KFE:RequestedBP:BeamClass_RBV', kind='omitted',
                            doc='Requested Beam Class')
    _reqBP_apply_cmd = Cpt(EpicsSignal, 'PMPS:KFE:BeamParamCntl:ReqBP:Apply', kind='omitted')
    transmission_scan = Cpt(TmoBeamLinePositioner, 'PMPS:KFE:BeamParamCntl:ReqBP:Transmission', kind='omitted')
    _transmission_cntl = Cpt(EpicsSignal, 'PMPS:KFE:BeamParamCntl:ReqBP:Transmission_RBV', write_pv='PMPS:KFE:BeamParamCntl:ReqBP:Transmission', kind='omitted',
                       doc='AT1K0 Transmission setting from beamline control.')
    
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
            if ( (transmission < 1) and self.pmps_override._apply_cmd.get() ):
                transmission = transmission * (self.pmps_override.max_eng.get()/5)
            self._transmission_cntl.put(transmission)
#        if beam_class is not None:
#            self.beamclass_request.put(beam_class)
        self._reqBP_apply_cmd.put(1)
    

    def beam_rate_watcher(self, stop=False):
        """
        Monitor the beam rate on the SXR line, and updates to the screen whenever there is a change.
        
        Parameters
        ----------
        stop : Bool, optional
            stop the watcher from updating to the screen. The default is False.
        """
        def _data_cb(*args, value, timestamp, **kwargs):
            rate_str = pyfiglet.figlet_format(str(round(value)) + " Hz")
            print('Beam Rate to SXR:\n' + rate_str)
            if value == 0:
                print('No Beam on SXR line!!!')
            return
        if not stop:
            uid=self.rate.subscribe(_data_cb)
        else:
            self.rate.unsubscribe_all

    def __init__(self, prefix='', name='beam_stats', **kwargs):
        super().__init__(prefix=prefix, name=name, **kwargs)
    
        if self.facility_mode.get() == 'NC':
            self.rate = Cpt(TmoBeamLinePositioner, 'PMPS:KFE:BeamParamCntl:ReqBP:Rate', kind='normal',
                            doc='LCLS BEAM Event Rate [Hz]')
    
tmo_beamline_cntl = TmoBeamLineControls(name="tmo_beamline_control")

class SXRGasAperture(BaseInterface, GroupDevice):
    tab_component_names = True  
    tab_whitelist = ['set_aperture']
    maa_01 = Cpt(GattApertureY, ':01:Y:STATE', kind='hinted')
    maa_02 = Cpt(GattApertureY, ':02:Y:STATE', kind='hinted')
    maa_03 = Cpt(GattApertureY, ':03:Y:STATE', kind='hinted')
    maa_04 = Cpt(GattApertureY, ':04:Y:STATE', kind='hinted')
    
    def set_aperture(self, state):
        self.maa_01.move(state)
        self.maa_02.move(state)
        self.maa_03.move(state)
        self.maa_04.move(state)
    
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
    
class UndSeg(Device):
    """
    Undulator Segement Device
    
    """
    k_act = UCpt(EpicsSignalRO)
    x_bpm = UCpt(EpicsSignalRO)
    y_bpm = UCpt(EpicsSignalRO)

    def __init__(self, prefix, **kwargs):
        UCpt.collect_prefixes(self, kwargs)
        super().__init__(prefix, **kwargs)

def _build_und_list(segs):
    defn = OrderedDict()
    for und_ind in segs:
        k_act_pv = 'USEG:UNDS:%d50:KAct' % (und_ind)
        x_bpm_pv = 'BPMS:UNDS:%d90:XSCS1H' % (und_ind)
        y_bpm_pv = 'BPMS:UNDS:%d90:YSCS1H' % (und_ind)
        name = 'u' + str(und_ind)
        defn[name.lower()] = Cpt(UndSeg, "", k_act_prefix=k_act_pv, 
                                 x_bpm_prefix=x_bpm_pv, 
                                 y_bpm_prefix=y_bpm_pv,
                                 kind='hinted')
    return defn
       
class UndulatorParameter(BaseInterface, GroupDevice):
    """
    Properties of Undulator Line
        k_act - K value of each undulator
        x_bpm - value from x-BPM 
        y_bpm - value from y-BPM
    """
    tab_component_names = True
    tab_whitelist = ['get_k','get_bpm']

    # def __init__(self):    
    #     for und_ind in u_segs:
    #         k_act_pv = 'USEG:UNDS:%d50:KAct' % (und_ind)
    #         x_bpm_pv = 'BPMS:UNDS:%d90:XSCS1H' % (und_ind)
    #         y_bpm_pv = 'BPMS:UNDS:%d90:YSCS1H' % (und_ind)
    #         name = 'u' + str(und_ind)
    #         setattr(self, name.lower(), UndSeg("",k_act_prefix=k_act_pv, x_bpm_prefix=x_bpm_pv, y_bpm_prefix=y_bpm_pv, name=name.lower()))
    def get_k(self, log=False, plot=False, msg=None, **kwargs):
        """
        Get current position of the undulator K.
        Parameters:
        -------------
        log - boolean to log the positions in the elog 
        msg - optional argument to append a message to the elog with the KB positions 
        plot - display a plot of the positions
        """
        curr_positions = {}
        segs = []
        for keys in dir(self):
            und = getattr(self, keys)
            if hasattr(und, 'k_act'):
                curr_positions[keys] = und.k_act.get()
                segs.append(int(keys[1:]))
        kwargs['tags'] = kwargs.get('tags', ' ') + ' undulator_k'        
        _log_positions(curr_positions, log, msg, **kwargs)          
        
        if plot:
            plt.figure()
            plt.plot(segs, list(curr_positions.values()))
            plt.ylabel('K-value')
            plt.xlabel('Undulator Segment')
            plt.show()

    def get_bpm(self, stop_update=False):
        """
        Get current beam position (x,y) from the bpm on each undulator section.
        """
        def _data_cb(*args, value, timestamp, obj, **kwargs):
            xy = obj.name[-5]
            und_num = obj.name[-8:-6]
            positions[xy][und_num] = value
            #ax.plot(u_segs, list(positions[xy].values()))
            lines[xy].set_ydata(list(positions[xy].values()))
            fig.canvas.draw()
            fig.canvas.flush_events()
        
        if stop_update:
            for key in dir(self):
                und = getattr(self, key)
                if hasattr(und, 'x_bpm'):
                    und.x_bpm.unsubscribe_all()
                if hasattr(und, 'y_bpm'):
                    und.y_bpm.unsubscribe_all()
            return

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

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        lx, = ax.plot(segs, list(positions['x'].values()))
        ly, = ax.plot(segs, list(positions['y'].values()))
        lines['x'] = lx
        lines['y'] = ly
        plt.ylabel('BPM')
        plt.xlabel('Undulator Segment')
        plt.draw()
       
        uid={}
        for key in dir(self):
            und = getattr(self, key)
            if hasattr(und, 'x_bpm'):
                uid[und.name+'x'] = und.x_bpm.subscribe(_data_cb)
            if hasattr(und, 'y_bpm'):
                uid[und.name+'y'] = und.y_bpm.subscribe(_data_cb)
        
#        print('ctrl+c to quit...')
        
        # try:
        #     while True:
        #         time.sleep(1)
        #         ax.plot(u_segs, list(positions['x'].values()))
        #         ax.plot(u_segs, list(positions['y'].values()))
        #         fig.canvas.draw()
        # except KeyboardInterrupt:
        # for key in dir(self):
        #     und = getattr(self, key)
        #     if hasattr(und, 'x_bpm'):
        #         und.x_bpm.unsubscribe(uid[und.name+'x'])
        #     if hasattr(und, 'y_bpm'):
        #         und.y_bpm.unsubscribe(uid[und.name+'y']) 

u_segs = list(range(26,35)) + list(range(36,48))
Unds = type('Unds',(UndulatorParameter, BaseInterface, Device), _build_und_list(u_segs))
unds = Unds(name='undulators')
#unds = UndulatorParameter()

def stop_xray():
    """
    This function puts st1k4 in to block the beam. 
    """
    st1k4.state('IN')
#    tmo_beamline_cntl.rate.mv(0)

# function to get SQ1 positions... consider if there is a better way to do this... JPC
def  get_cvmi_sq1(log=False, msg=None, **kwargs):
    """
    Get current position of the cVMI sq1 stand.
    Parameters:
    ---------------
    log - boolean to log the positions in the elog 
    msg - optional argument to append a message to the elog with the KB positions 
    """
    curr_positions= {
    "sq1 x":caget("MCQ:TMO:02:TARGET:X:RBV"),
    "sq1_y":caget("MCQ:TMO:02:TARGET:Y:RBV"),
    "sq1_z":caget("MCQ:TMO:02:TARGET:Z:RBV"),
    "sq1_rx":caget("MCQ:TMO:02:TARGET:rX:RBV"),
    "sq1_ry":caget("MCQ:TMO:02:TARGET:rY:RBV"), 
    "sq1_rz":caget("MCQ:TMO:02:TARGET:rZ:RBV"),
    }
    kwargs['tags'] = kwargs.get('tags', ' ') + ' sq1'        
    _log_positions(curr_positions, log, msg, **kwargs)

def get_cvmi_motion(log=False, msg=None, **kwargs):
    """
    Get current position of the cVMI motion.
    Parameters:
    -------------
    log - boolean to log the positions in the elog 
    msg - optional argument to append a message to the elog with the KB positions 
    
    """
    curr_positions = {}
    for keys in dir(tmo_cvmi):
        motor = getattr(tmo_cvmi, keys)
        if hasattr(motor, 'position'):
            curr_positions[keys] = getattr(motor, 'position')
    kwargs['tags'] = kwargs.get('tags', ' ') + ' cvmi_positions'        
    _log_positions(curr_positions, log, msg, **kwargs)

def  get_lamp_sq1(log=False, msg=None, **kwargs):
    """
    Get current position of the cVMI sq1 stand.
    Parameters:
    ---------------
    log - boolean to log the positions in the elog 
    msg - optional argument to append a message to the elog with the KB positions 
    """
    curr_positions= {
    "sq1 x":caget("MCQ:TMO:01:TARGET:X:RBV"),
    "sq1_y":caget("MCQ:TMO:01:TARGET:Y:RBV"),
    "sq1_z":caget("MCQ:TMO:01:TARGET:Z:RBV"),
    "sq1_rx":caget("MCQ:TMO:01:TARGET:rX:RBV"),
    "sq1_ry":caget("MCQ:TMO:01:TARGET:rY:RBV"), 
    "sq1_rz":caget("MCQ:TMO:01:TARGET:rZ:RBV"),
    }
    kwargs['tags'] = kwargs.get('tags', ' ') + ' sq1'        
    _log_positions(curr_positions, log, msg, **kwargs)
    
def get_lamp_motion(log=False, msg=None, **kwargs):
    """
    Get current position of the LAMP motion.
    Parameters:
    -------------
    log - boolean to log the positions in the elog 
    msg - optional argument to append a message to the elog with the KB positions 
    
    """
    curr_positions = {}
    for keys in dir(tmo_lamp):
        motor = getattr(tmo_lamp, keys)
        if hasattr(motor, 'connected'):
            if motor.connected:
                if hasattr(motor, 'position'):
                    curr_positions[keys] = getattr(motor, 'position')
    kwargs['tags'] = kwargs.get('tags', ' ') + ' lamp_positions'        
    _log_positions(curr_positions, log, msg, **kwargs)

def get_mrco_motion(log=False, msg=None, **kwargs):
    """
    Get current position of the MRCO motion.
    Parameters:
    -------------
    log - boolean to log the positions in the elog 
    msg - optional argument to append a message to the elog with the KB positions 
    
    """
    curr_positions = {}
    for keys in dir(tmo_mrco):
        motor = getattr(tmo_mrco, keys)
        if hasattr(motor, 'connected'):
            if motor.connected:
                if hasattr(motor, 'position'):
                    curr_positions[keys] = getattr(motor, 'position')
    kwargs['tags'] = kwargs.get('tags', ' ') + ' mrco_positions'        
    _log_positions(curr_positions, log, msg, **kwargs)    
    
def mv_to_preset(device, preset):
    """
    Move the 'device' to a preset position described by 'preset'.
    Parameters
    ----------
    device : object with multiple axes
        Device with multiple axes which you want to move to a preset location
    preset : str.
        Name of preset position.

    Returns
    -------
    None.

    """
    for keys in dir(device):
        motor = getattr(device, keys)
        if hasattr(motor, 'presets'):
            if (motor.presets.has_presets):
                if (preset in motor.presets.positions.__dict__):
                    move_to_preset = getattr(motor, 'mv_'+preset )
                    move_to_preset
    
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

def toggle_arb(device, log=True):
    """
    Toggle the arbitration state from True to False and back to True for the device.

    Parameters
    ----------
    device : device with atribute target
        Device to have arbitration toggled.
    log : bool, optional
        Write to elog that arbitration has been toggled. The default is True.

    """
    if hasattr(device, 'target'):
        target = getattr(device, 'target')
    elif hasattr(device, 'mirror'):
        target = getattr(device, 'mirror')
    elif hasattr(device, 'coating'):
        target = getattr(device, 'coating')    
    else:
        logger.error(device.name + ' does not have attribute "target".')
        return      

    if not hasattr(target, 'arb_enable'):
        logger.error(target.name + ' does not have attribute "arb_enable".')
    else:
        target.arb_enable.put(0)
        time.sleep(2)
        target.arb_enable.put(1)
        if log:
            elog.post('toggled arbitration status of '+target.name, tags=(device.name+' PMPS_toggle'))
    

def get_PM(imager, duration=60, log=True, filename=None, notes=None):
    """
    Measure the power meter 

    Parameters
    ----------
    imager : PPM Imager
        Imager to use for power meter measurement.
    duration : int, optional
        Time to integrate in the measurement. The default is 60. 
    log : bool, optional
        Flag for saving data to h5 file. The default is True.
    filename : str, optional
        name for h5 file to save data. The default is to save with the imager name and time stamp.
        data is saved to /cds/home/opr/tmoopr/experiments/ppm_data/
    notes : str, optional
        notes to append to an elog post. The default is None
        
    """
#    _responsivity = {'im2k4':0.0712, 'im3k4':0.0751, 'im4k4':0.0721, 'im5k4':0.0862} #From Phil H. on 5/9/24 units are V/W
    def _data_cb(value, timestamp, **kwargs):
        obj = kwargs["obj"]
        if isinstance(value, np.ndarray):
            data[obj.name].extend(value)
        else:
            data[obj.name].append(value)

#    def _data_cb_PPM(*args, value, timestamp, **kwargs):
#        data['PPM'].extend(value)
#        return
    
#    def _data_cb_gmd(*args, value, timestamp, obj, **kwargs):
#        data[obj.name].append(value)        
#        return

    if ('PPM' not in type(imager).__name__) :
        logger.error('imager has no power meter')
        return
    
    imager_pos = imager.target()
    init_rate = tmo_beamline_cntl.rate.get()
    if imager_pos != 'POWERMETER':    
        imager.target('POWERMETER', wait=True)

    if tmo_beamline_cntl.rate.get()<(0.9*init_rate):
        logger.info('Waiting for rate....')
        while tmo_beamline_cntl.rate.get() < (0.9*init_rate):
            time.sleep(1)
        
    devices = [gmd.mj, xgmd.mj, gmd.avg_int, xgmd.avg_int, imager.power_meter.raw_voltage_buffer]
    
    logger.info('Starting PPM collection for ' + imager.name)    
    data={}
#    data['PPM'] = []
#    uid = imager.power_meter.raw_voltage_buffer.subscribe(_data_cb_PPM)
#    uid = imager.power_meter.raw_voltage_buffer.subscribe(_data_cb)
    uid = {}
    for dev in devices:
        data[dev.name] = []
#        uidd[dev.name] = dev.subscribe(_data_cb_gmd)
        uid[dev.name] = dev.subscribe(_data_cb)
        
    time.sleep(duration)  
    logger.info('Done collecting PPM data')
    
#    imager.power_meter.raw_voltage_buffer.unsubscribe(uid)
    for dev in devices:
        dev.unsubscribe(uid[dev.name])

    result = {}
    for key in data:
        result[key] = sum(data[key])/len(data[key])
    
    measurementDF = pd.DataFrame({"Measurement:": result.keys(), "Average:": result.values()})
    print (measurementDF)
    params=['rate', 'photon energy', 'transmission']
    values=[tmo_beamline_cntl.rate.get(), tmo_beamline_cntl.photon_eV.get(),tmo_beamline_cntl.transmission.get()]
    blDF = pd.DataFrame({"Parameter:": params, "Value:": values})
    
    imager.target('OUT')

    if log:
        if not filename:
            filename = imager.name + '-' + str(int(time.time()))+'.h5'
        PPM_PATH='/cds/home/opr/tmoopr/experiments/ppm_data/'
        location = ''.join([PPM_PATH, filename])
        hf = h5py.File(location, 'w')
        for key in data:
            hf.create_dataset(key+'_data', data=data[key])
        hf.close()
        logger.info(f'wrote all data to {location}')
        msg = imager.name + ' power meter measurement saved to: \n' + location 
        if notes:
            msg = msg + '\n\n' + notes + '\n'
        msg = msg + '\n'+ measurementDF.to_string() + '\n' + 'Beamline Parameters:\n' + blDF.to_string()
        
        tag = imager.name+' powermeter'
        elog.post(msg, tags=tag)

    return

def calibrate_PM(imager, duration=60, notes=None):
    """
    Calibrate the voltage to average power conversion for a PPM Imager.

    Parameters
    ----------
    imager : PPM Imager
        Imager to use for power meter measurement.
    duration : int, optional
        Time to integrate in the measurement. The default is 60. 
    notes : str, optional
        notes to append to an elog post. The default is None
        
    """
    def _data_cb(value, timestamp, **kwargs):
        obj = kwargs["obj"]
        if isinstance(value, np.ndarray):
            data[obj.name].extend(value)
        else:
            data[obj.name].append(value)

    if ('PPM' not in type(imager).__name__) :
        logger.error('imager has no power meter')
        return
    
    ori_imager_pos = imager.target()
    init_rate = tmo_beamline_cntl.rate.get()
    im2k0.target('YAG')
    if ori_imager_pos != 'POWERMETER':    
        imager.target('POWERMETER', wait=True)
    
    data={}   
    data[imager.power_meter.raw_voltage_buffer.name]=[]
    uid = imager.power_meter.raw_voltage_buffer.subscribe(_data_cb)    
    logger.info('Collecting PPM background for ' + imager.name)
    time.sleep(10)
    imager.power_meter.raw_voltage_buffer.unsubscribe(uid)
    logger.info('Done collecting Offset data')
    im2k0.target('OUT')
    
    offset = sum(data[imager.power_meter.raw_voltage_buffer.name])/len(data[imager.power_meter.raw_voltage_buffer.name])

    devices = [gmd.mj, xgmd.mj, gmd.avg_int, xgmd.avg_int, imager.power_meter.raw_voltage_buffer]

    if tmo_beamline_cntl.rate.get()<(0.9*init_rate):
        logger.info('Waiting for rate....')
        while tmo_beamline_cntl.rate.get() < (0.9*init_rate):
            time.sleep(1)
    
    logger.info('Starting PPM collection for ' + imager.name)    
    uid = {}
    for dev in devices:
        data[dev.name] = []
        uid[dev.name] = dev.subscribe(_data_cb)
        
    time.sleep(duration)  
    logger.info('Done collecting PPM calibration data')
    
    for dev in devices:
        dev.unsubscribe(uid[dev.name])
        
    result = {}
    for key in data:
        result[key] = sum(data[key])/len(data[key])

    old_params=OrderedDict()
    old_params[imager.power_meter.calib_offset.name] = imager.power_meter.calib_offset.get()
    old_params[imager.power_meter.calib_ratio.name] = imager.power_meter.calib_ratio.get()
    old_params[imager.power_meter.calib_mj_ratio.name] = imager.power_meter.calib_mj_ratio.get()
    
    imager.power_meter.calib_offset.put(-offset)
    imager.power_meter.calib_ratio.put(1)
    imager.power_meter.calib_mj_ratio.put(result[gmd.avg_int.name] / (result[imager.power_meter.raw_voltage_buffer.name] -offset))
    
    time.sleep(2)
    
    new_params=OrderedDict()
    new_params[imager.power_meter.calib_offset.name] = imager.power_meter.calib_offset.get()
    new_params[imager.power_meter.calib_ratio.name] = imager.power_meter.calib_ratio.get()
    new_params[imager.power_meter.calib_mj_ratio.name] = imager.power_meter.calib_mj_ratio.get()

    paramDF = pd.DataFrame({"Parameter:": new_params.keys(), 
                            "Old Value:": old_params.values(), 
                            "New Value:": new_params.values()})  
    
    msg = imager.name+' powermeter calibration.'
    if notes:
        msg = msg + '\n\n' + notes + '\n'
    msg = msg + '\n'+ paramDF.to_string()
    
    tag = imager.name+' powermeter_calibration'
    elog.post(msg, tags=tag)
    logger.info('Calibration Parameters Updated')
    
    imager.target(ori_imager_pos)
    
    return