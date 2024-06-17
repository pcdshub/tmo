from hutch_python.utils import safe_load
import numpy as np
import matplotlib.pyplot as plt
import sys
import time as time
from epics import caget, caput, cainfo
#from elog import HutchELog as elog_temp
from tmo.db import elog
import pandas as pd
sys.path.append('/cds/home/opr/tmoopr')

#import las_scan_TW as LAS

with safe_load('TMO MPOD Utilities'):
    from tmo.tmo_mpods import get_mpod, tmo_mpod_off, tmo_mpod #, mpod_modules
#from tmo.tmo_mpods import tmo_mpod

with safe_load('Beam Energy Request'):
#    from pcdsdevices.beam_stats import BeamEnergyRequest
    from tmo.tmo_photon_energy import hf_w, hf_2w, hf_w2w, hf_w_status, hf_2w_status
#    photon_energy1 = BeamEnergyRequest('TMO',name='photon_energy1',atol=0,skip_small_moves=False,bunch=1)
#    photon_energy2 = BeamEnergyRequest('TMO',name='photon_energy2',atol=0,skip_small_moves=False,bunch=2)

with safe_load('Vacuum Checkout'):
    from tmo.vac_checkout import tmo_vac_checkout 
    from tmo.tmo_hvac import tmo_hvac, tmo_hvac_lamp, tmo_hvac_mrco

#with safe_load('TMO Scans'):
#    from tmo.tmo_scans import move_and_wait, imprint_scan, las_scan_JPC

#with safe_load('TMO Sequencers'):
#    from tmo.tmo_sequencer import tmo_sequencer1, tmo_sequencer8, tmo_sequencer9, gas_jet_sequence, goose_trigger, burst_trigger, get_tmo_sequencer_status, tmo_evr0, tmo_evr1, tmo_evr2
#    import tmo.tmo_sequencer

#with safe_load('END BEAMTIME'):
#    from tmo.end_beamTime import transmission_set, transmission_set_fraction, andor_on_off, end_of_beamtime, imagers_out, st4k4_in_out

#with safe_load('TMO Beamline Controls'):
#    from tmo.tmo_beamline_control import stop_xray 
    
with safe_load('User Sequences'):
    import tmo.user_seq as user_seq

with safe_load('BYKIKS'):
    from tmo.bykiks import bykiks, set_bykik, get_bykik

with safe_load('CVMI Utilities'):
    from tmo.tmo_beamline_control import get_cvmi_sq1, get_cvmi_motion

with safe_load('LAMP Utilities'):
    from tmo.tmo_beamline_control import get_lamp_sq1, get_lamp_motion

with safe_load('MRCO Utilities'):
    from tmo.tmo_beamline_control import get_mrco_motion
    
with safe_load('Undulator Utilities'):
    from tmo.tmo_beamline_control import unds, get_unds 
    
with safe_load('Beamline Utilities'):
    from tmo.tmo_beamline_control import get_PM, get_rtds, set_rtds, get_vac_valves, stop_xray, toggle_arb, calibrate_PM

with safe_load('TMO Beamline Control'):
    from tmo.tmo_beamline_control import tmo_beamline_cntl
        
with safe_load('Missing Beamline Components'):
    from tmo.tmo_beamline_control import gmd, xgmd, at1k0    

with safe_load('TMO MBES Utility'):
    from tmo.tmo_mbes_utility import mbes_voltage, tmo_lamp_vcn

with safe_load('CVMI Utilities'):
    from tmo.tmo_cvmi_utility import cvmi_voltage_utility, cvmi_optics

with safe_load('FS14 lxt, txt, and lxt_ttc'):
    from tmo.lxt2 import lxt, txt, lxt_ttc, shift_t0, get_timing

with safe_load('CAM Recorder'):
    from tmo.cam_to_file import h5_img_collect, ppm_scan

with safe_load('CAM Fiducial Detection'):
    from tmo.cam_fiducial_detection import collect_fiducial_image, analyze_image, check_fiducial_change, check_fiducial_change_orb

with safe_load('TMO OPTICS UTILITIES'):
    from tmo.tmo_optics import get_KB1, set_KB1, get_fzp, get_mr1k4, get_KB2, get_slits, fix_wfs, reset_slits

with safe_load('TMO Imager Utility'):
    from tmo.tmo_imager_align import tmo_imagers

with safe_load('TMO Laser'):
    from tmo.tmo_laser import tmo_laser, get_lm1k4, get_lm2k4, get_atm, stop_laser, get_laser_tpr

with safe_load('LCLS-II get_daq - Timed runs and scans'):
    from tmo.tmo_daq_scans import scan, daq_stats
   
with safe_load('continous scan'):
    from tmo.continuous_scan import continuous_scan
    
with safe_load('KB Transmission Check'):
    from tmo.mirror_check import check_kbo1_transmission, check_kbo2_transmission

def log_change(motor, daq=None):
    """
    This function logs changes of a motor position to the elog. If the function is exceuted during a run, the post will append to the current run.
    motor: motor object to log change
    daq: optional arguement. Pass the daq object if you want the log to be associated with the current run number.   
    """
    motor_name = motor.name
    if (daq == None):
        record_state = False
    else:
        record_state = daq.control.getStatus()[3]
        run_num = daq.control.getStatus()[-2]
    if hasattr(motor, 'position'):
        new_value = motor.position
    else:
        print("Motor should have position attribute for this method. Trying .get(), fingers crossed you get something useable.")
        new_value = motor.get()
    msg = "shifted " + motor_name + " to " + str(new_value)
    if record_state:
        elog.post(msg, run=run_num)
    else:
        elog.post(msg)

# from pcdsdevices.sqr1 import SQR1
with safe_load('SQ1'):
    from tmo.sqr1 import SQR1
    tmo_lamp_sq1 = SQR1('MCQ:TMO:01', name='tmo_lamp_sq1')
