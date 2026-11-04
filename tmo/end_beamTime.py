import numpy as np
import time
from tmo.db import *
from epics import caget, caput, cainfo
from pcdsdevices.epics_motor import EpicsMotorInterface
from ophyd import EpicsSignal, EpicsSignalRO
from tmo.tmo_beamline_control import tmo_beamline_cntl, set_rtds, get_stprs
from tmo.tmo_hvac import tmo_hvac
from tmo.tmo_optics import get_slits, get_KB1, get_KB2, get_fzp, get_mr1k4, sp1k4_intensifier
from tmo.tmo_imager_align import tmo_imagers
from tmo.tmo_mpods import tmo_mpod, get_mpod
from tmo.tmo_pdu import tmo_pdu
from tmo.tmo_beckoff import tmo_beckoff
#from tmo.tmo_mbes_utility import ortec_ps
from tmo.tmo_dream_utils import tmo_dream_coil
from tmo.tmo_endstation import get_mrco_motion, get_lamp_motion, get_dream_motion, get_mbes_motion
from tmo.tmo_endstation import tmo_mrco_vcn, get_mrco_vcn, tmo_lamp_vcn, get_lamp_vcn, tmo_mbes_vcn, get_mbes_vcn
from tmo.lxt2 import get_timing, get_laser_tpr
import logging
from pcdsdevices.slits import SL2K4Slits as sl2k4
logger = logging.getLogger(__name__)

# def transmission_set(percentage):
# 	'''Sets the gas attenuator value in percentage of the transmission. Works only in PMPS mode.
# 	Usage: transmission_set(percentage). Example: transmission_set(10) sets to 10% transmission'''
# 	fract = percentage/100.0
# 	if fract > 1.0:
# 		print('Transmission can not be more 100%. Try a lower value')
# 		return False
#         pmps_override_energy = tmo_beamline_cntl.pmps_override_max_eng.get()
#         pmps_SC_MAX = 5.0
#         tmo_beamline_cntl.transmission_scan.mv(fract*pmps_override_energy/pmps_SC_MAX)
# 	time.sleep(1)

# def transmission_set_fraction(fract):
# 	'''Sets the gas attenuator value in fraction of the transmission. Works only in PMPS mode.
# 	Usage: transmission_set_fraction(fraction). Example: transmission_set(.1) sets to 10% transmission'''
# 	if fract > 1.0:
# 		print('Transmission can not be more 100%. Try a lower value.')
# 		return False
#         pmps_override_energy = tmo_beamline_cntl.pmps_override_max_eng.get()
#         pmps_SC_MAX = 5.0
#         tmo_beamline_cntl.transmission_scan.mv(fract*pmps_override_energy/pmps_SC_MAX)
# 	time.sleep(1)

# def rate_change(rates):
# 	'''Rates can only be 120, 10, 1 or 0 Hz'''
# 	if rates in [0, 1, 10, 120]:
# 		tmo_beamline_cntl.rate.mv(rates)
# 	else:
# 		print('Rates must be either of 0, 1, 10, or 120 Hz')
# 	time.sleep(0.5)
        

# def st4k4_in_out(state=0):
# 	'''ST4K4 stopper movement function. 
# 	usage: st4k4_in_out(state=0)		
# 	Set the state to 0 for insertion.
# 	For retraction, set the state to 1.'''
# 	if (caget('ST4K4:TMO_TERM:CMD:OUT') == state):
# 		if state == 0:
# 			print('ST4K4 is now inserted')
# 		else:
# 			print('ST4K4 is now retracted')
# 	else:
# 		caput('ST4K4:TMO_TERM:CMD:OUT', state)
# 		if state == 0:
# 			print('ST4K4 is now inserted')
# 		else:
# 			print('ST4K4 is now retracted')

# def andor_on_off(state='ON'):
# 	if state=='ON':
# 		caput('TMO:VLS:CAM:01:AndorCooler', 1)
# 	else: 
# 		caput('TMO:VLS:CAM:01:AndorCooler', 0)
# 		caput('ST4K4:TMO_TERM:CMD:OUT', 0)
# 		caput('VLS:VAC:SETUP_MODE', 0)


# insertion_device_list = [im1k4, im2k4, im3k4, im4k4, im5k4, im6k4, pf1k4, pf2k4, tm1k4, tm2k4]

# def imagers_out():
# 	for device in insertion_device_list:
# 		device.target('OUT')

# def arb_enable_set(arb_enable_state=True):
# 	for device in insertion_device_list:
# 		device.target.arb_enable.put(arb_enable_state) 
	

def end_of_beamtime(skip_pmps=False):
    '''This function closes all K4 valves, and set transmission to 100%, stops the Goose Trigger. MUST have the beam stopper in for this function to work.
    	Usage: end_of_beamtime()

	'''
    amp_ps_2 = EpicsSignal('TMO:USR:BHC:BOUT:1')
    
    logger.info('Logging FZP positions...')
    get_fzp(log=True, msg='FZP positions at end of shift', include_cam=['sp1k4-gige'], tags='script-generated')
    logger.info('Removing FZP paddles...')
    sp1k4.zone_plate('OUT')
    sp1k4.solid_att('OUT')
    if sp1k4_intensifier.mcp_voltage.connected:
        logger.info('Disable sp1k4 intensifier...')
        sp1k4_intensifier.mcp_voltage.put(450, wait=True) 
        sp1k4_intensifier.disable() # on 20250315 TD commented this out (then back in) because the PV was timing out
        time.sleep(2)
    
#    logger.info('Closing Stoppers...')
#    st4k4.insert_signal.put(1)
#    st1k4.state('IN')
#    time.sleep(5)
    get_stprs()
#    st3k4_pps.in_signal.unsubscribe_all()
    time.sleep(2)
    
    logger.info('Closing vacuum gate valves....')
    tmo_hvac.close_valves()
    time.sleep(2)
    
    logger.info('Logging slit positions...')
    get_slits([sl2k0,sl1k4], log=True, msg='Slit positions at the end of shift.')
    set_rtds(state=0)
    time.sleep(2)
    
    logger.info('Logging KB positions......')
    get_mr1k4(log=True, msg='SOMS at end of shift.', tags='script-generated')
    get_KB1(log=True, msg='KB1 positions at end of shift.', tags='script-generated')
    get_KB2(log=True, msg='KB2 positions at end of shift.', tags='script-generated')
    time.sleep(2)
    
    logger.info('Clearing paddles from beam path...')
    tmo_imagers.all_out()
    pf1k4.target('OUT')
    pf2k4.target('OUT')
    time.sleep(2)
    
    logger.info('Logging HV...')
    get_mpod(log=True, msg='HV settings at end of shift', tags='script-generated')
    logger.info('Shutting off HV...')
    
    if tmo_dream.connected: #Turn off DREAM amplifiers
        tmo_dream_coil.off()
        for c_name in dir(tmo_pdu.dream_pdu):
            ch = getattr(tmo_pdu.dream_pdu, c_name)
            if hasattr(ch, 'desc'):
                if 'ARR-15' in ch.desc.get():
                    ch.off()
                    logger.info('%s PS off' % ch.desc.get())
            
    if tmo_lamp.connected: #Turn off LAMP Amplifiers
        for c_name in dir(tmo_pdu.lamp_pdu):
            ch = getattr(tmo_pdu.lamp_pdu, c_name)
            if hasattr(ch, 'desc'):
                if 'ortec' in ch.desc.get():
                    ch.off()
                    logger.info('%s PS off' % ch.desc.get())
    
    if tmo_mrco.connected: #Turn off MRCO amplifiers 
        for i in range(1,9):
            ch = getattr(tmo_beckoff, 'dout_%d'%i)
            if 'Amplifiers' in ch.desc.get():
                ch.off()
                logger.info('%s PS off' % ch.desc.get())
    
    if tmo_mbes.connected:
        for c_name in dir(tmo_pdu.cvmi_pdu):
            ch = getattr(tmo_pdu.cvmi_pdu, c_name)
            if hasattr(ch, 'desc'):
                if 'ortec' in ch.desc.get():
                    ch.off()
                    logger.info('%s PS off' % ch.desc.get())
        
    #stop the MPOD.
    tmo_mpod.all_off()
    for ch_name in dir(tmo_mpod):
        ch = getattr(tmo_mpod, ch_name)
        if hasattr(ch, 'set_voltage'):
            ch.set_voltage(0)
    time.sleep(2)
          
    logger.info('Cancel PMPS override...')
    if skip_pmps:
        logger.info('Skip PMPS\n')
    else:
        tmo_beamline_cntl.pmps_override.cancel()
        tmo_beamline_cntl.apply_beam_parameter(beam_class=15, transmission=1)
    time.sleep(2)
    
    logger.info('Logging laser timing...')
    get_timing(log=True, msg='Laser timing at end of shift', tags='script-generated')
    time.sleep(1)
    
    logger.info('Logging laser triggering...')
    get_laser_tpr(log=True, msg='IP1 laser triggering at end of shift.', tags='script-generated')
    
    logger.info('Adding all devices to arbitration...')
    from tmo.db import m
    for dev_name in dir(m):
        dev = getattr(m,dev_name)
        if hasattr(dev,  'arb_enable'):
            dev.arb_enable.put(1)
    time.sleep(2)
    
    if daq.daqState == 'running':        
        logger.info('Stopping DAQ...')
        daq._set_connected()
    time.sleep(2)

    logger.info('Logging IP1 endstation positions...')
    if tmo_lamp.connected:
        logger.info('Logging LAMP positions')
        get_lamp_motion(log=True, msg='LAMP positions at end of shift.', include_cam=['tmo-07','lamp-01'], tags='script-generated')
    if tmo_mrco.connected:
        logger.info('Logging MRCO positions')
        get_mrco_motion(log=True, msg='MRCO positions at end of shift.', tags='script-generated')
    if tmo_mbes.connected:
        logger.info('logging MBES positions')
        get_mbes_motion(log=True, msg='MBES positions at end of shift.', tags='script-generated')
    if tmo_dream.connected:
        logger.info('Logging DREAM positions')
        get_dream_motion(log=True, msg='DREAM positions at end of shift.', tags='script-generated')
    time.sleep(2)
    
    logger.info('Stopping sample delivery, closing gas needle...')
    if tmo_lamp_vcn.connected:
        get_lamp_vcn(log=True, msg='LAMP VCN at end of shift', tags='script-generated')
        tmo_lamp_vcn.close_valve()
    if tmo_mrco_vcn.connected:
        get_mrco_vcn(log=True, msg='MRCO VCN at end of shift', tags='script-generated')
        tmo_mrco_vcn.close_valve()
    if tmo_mbes_vcn.connected:
        get_mbes_vcn(log=True, msg='MBES VCN at end of shift', tags='script-generated')
        tmo_mbes_vcn.close_valve()
#    if tmo_dream.connected:
#        tmo_beckoff.dout_2.off() #stop nozzle A delivery for DREAM
#        tmo_beckoff.dout_3.off() #stop nozzle B delivery for DREAM
    
    
    
    
    
    
#     beam_rate = 120
# 	fzp_out_X = -80
# 	fzp_out_Y = -7.7
# 	now = datetime.now()

# 	try:
# 		stopper = st3k4_pps._inserted
# 		while not stopper:
# 			input("Stopper is not in, put in stopper and press enter?\n")
# 			stopper = st3k4_pps._inserted
# 	except:
# 		print('Did not find stopper state')
# 	
# 	imagers_out()
# 	arb_enable_set()
# 	time.sleep(0.5)	
# 	print('All imagers are out')


# 	st4k4_in_out(0)
# 	tmo_hvac.close_valves()
# 	print('All K4 line valves are closed.')

# 	transmission_set(100)
# 	#rtdsk0.mpa1.remove()
# 	#rtdsk0.mpa2.remove()
# 	set_rtds(state=0)
# 	print('Transmission set to 100%')

# 	rate_change(beam_rate)
# 	print('Beam rate set to 120 Hz')

# 	goose_trigger(1,0)
# 	print('Goose trigger turned off')

# 	tmo_sequencer1.stop()
# 	print('EL valve sequencer stopped')

# 	tm1k4.target('OUT')
# 	print('Moved ATM sample paddle to OUT position')

# 	sp1k4.zone_plate_x.mv(fzp_out_X)
# 	sp1k4.zone_plate_y.mv(fzp_out_Y)
# 	waitTime = 6*(sp1k4.zone_plate_x.wm() - fzp_out_X)
# 	print('FZP spectrometer is set to OUT position. Wait for ' +str(waitTime)+' s ...')
# 	
# 	try:
# 		get_mr1k4(log=True, msg='MR1K4 position at the end of the shift on ' + now.strftime("%B-%d-%Y")) 
# 	except:
# 		print('Error getting position value from MR1K4')

# 	try:
# 		get_KB1(log=True, msg='KB1 position at the end of the shift on ' + now.strftime("%B-%d-%Y")) 
# 	except:
# 		print('Error getting position value from KB1')

# 	try:
# 		get_KB2(log=True, msg='KB2 position at the end of the shift on ' + now.strftime("%B-%d-%Y")) 
# 	except:
# 		print('Error getting position value from KB2')
# 	try:
# 		get_slits(slits=sl1k0, log=True, msg='SL1K0 position at the end of the shift on ' + now.strftime("%B-%d-%Y"))
# 		get_slits(slits=sl2k0, log=True, msg='SL2K0 position at the end of the shift on ' + now.strftime("%B-%d-%Y"))
# 		get_slits(slits=sl1k4, log=True, msg='SL1K4 position at the end of the shift on ' + now.strftime("%B-%d-%Y"))
# 		get_slits(slits=sl2k4, log=True, msg='SL2K4 position at the end of the shift on ' + now.strftime("%B-%d-%Y"))
# 	except:
# 		print('Error getting all the slits positions')
# 	print('Key component motor positions are logged in the current experiment logbook')

# 	#bringDownAll()
# 	#print('Bring all HVs down. Wait for 10s...')
# 	#time.sleep(10)
# 	#print('HVs all off.')
