import numpy as np

from tmo.db import *
from epics import caget, caput, cainfo
import time
from pcdsdevices.epics_motor import EpicsMotorInterface
from elog import HutchELog as elog_temp
elog = elog_temp.from_registry()
from tmo.tmo_sequencer import tmo_sequencer1, tmo_sequencer8, gas_jet_sequence, goose_trigger
from tmo.tmo_beamline_control import tmo_beamline_cntl, set_rtds
from tmo.vac_checkout import tmo_hvac
from tmo.tmo_optics import *
from datetime import datetime
from magBottle_utility import bringDownAll

def transmission_set(percentage):
	'''Sets the gas attenuator value in percentage of the transmission. Works only in PMPS mode.
	Usage: transmission_set(percentage). Example: transmission_set(10) sets to 10% transmission'''
	fract = percentage/100.0
	if fract > 1.0:
		print('Transmission can not be more 100%. Try a lower value')
		return False
        pmps_override_energy = tmo_beamline_cntl.pmps_override_max_eng.get()
        pmps_SC_MAX = 5.0
        tmo_beamline_cntl.transmission_scan.mv(fract*pmps_override_energy/pmps_SC_MAX)
	time.sleep(1)

def transmission_set_fraction(fract):
	'''Sets the gas attenuator value in fraction of the transmission. Works only in PMPS mode.
	Usage: transmission_set_fraction(fraction). Example: transmission_set(.1) sets to 10% transmission'''
	if fract > 1.0:
		print('Transmission can not be more 100%. Try a lower value.')
		return False
        pmps_override_energy = tmo_beamline_cntl.pmps_override_max_eng.get()
        pmps_SC_MAX = 5.0
        tmo_beamline_cntl.transmission_scan.mv(fract*pmps_override_energy/pmps_SC_MAX)
	time.sleep(1)

def rate_change(rates):
	'''Rates can only be 120, 10, 1 or 0 Hz'''
	if rates in [0, 1, 10, 120]:
		tmo_beamline_cntl.rate.mv(rates)
	else:
		print('Rates must be either of 0, 1, 10, or 120 Hz')
	time.sleep(0.5)
        

def st4k4_in_out(state=0):
	'''ST4K4 stopper movement function. 
	usage: st4k4_in_out(state=0)		
	Set the state to 0 for insertion.
	For retraction, set the state to 1.'''
	if (caget('ST4K4:TMO_TERM:CMD:OUT') == state):
		if state == 0:
			print('ST4K4 is now inserted')
		else:
			print('ST4K4 is now retracted')
	else:
		caput('ST4K4:TMO_TERM:CMD:OUT', state)
		if state == 0:
			print('ST4K4 is now inserted')
		else:
			print('ST4K4 is now retracted')

def andor_on_off(state='ON'):
	if state=='ON':
		caput('TMO:VLS:CAM:01:AndorCooler', 1)
	else: 
		caput('TMO:VLS:CAM:01:AndorCooler', 0)
		caput('ST4K4:TMO_TERM:CMD:OUT', 0)
		caput('VLS:VAC:SETUP_MODE', 0)


insertion_device_list = [im1k4, im2k4, im3k4, im4k4, im5k4, im6k4, pf1k4, pf2k4, tm1k4, tm2k4]

def imagers_out():
	for device in insertion_device_list:
		device.target('OUT')

def arb_enable_set(arb_enable_state=True):
	for device in insertion_device_list:
		device.target.arb_enable.put(arb_enable_state) 
	

def end_of_beamtime():
	'''This function closes all K4 valves, stops EL valve trigger, and set transmission to 100%, stops the Goose Trigger. MUST have the beam stopper in for this function to work.
	Usage: end_of_beamtime()

	'''
	beam_rate = 120
	fzp_out_X = -80
	fzp_out_Y = -7.7
	now = datetime.now()

	try:
		stopper = st3k4_pps._inserted
		while not stopper:
			input("Stopper is not in, put in stopper and press enter?\n")
			stopper = st3k4_pps._inserted
	except:
		print('Did not find stopper state')
	
	imagers_out()
	arb_enable_set()
	time.sleep(0.5)	
	print('All imagers are out')


	st4k4_in_out(0)
	tmo_hvac.close_valves()
	print('All K4 line valves are closed.')

	transmission_set(100)
	#rtdsk0.mpa1.remove()
	#rtdsk0.mpa2.remove()
	set_rtds(state=0)
	print('Transmission set to 100%')

	rate_change(beam_rate)
	print('Beam rate set to 120 Hz')

	goose_trigger(1,0)
	print('Goose trigger turned off')

	tmo_sequencer1.stop()
	print('EL valve sequencer stopped')

	tm1k4.target('OUT')
	print('Moved ATM sample paddle to OUT position')

	sp1k4.zone_plate_x.mv(fzp_out_X)
	sp1k4.zone_plate_y.mv(fzp_out_Y)
	waitTime = 6*(sp1k4.zone_plate_x.wm() - fzp_out_X)
	print('FZP spectrometer is set to OUT position. Wait for ' +str(waitTime)+' s ...')
	
	try:
		get_mr1k4(log=True, msg='MR1K4 position at the end of the shift on ' + now.strftime("%B-%d-%Y")) 
	except:
		print('Error getting position value from MR1K4')

	try:
		get_KB1(log=True, msg='KB1 position at the end of the shift on ' + now.strftime("%B-%d-%Y")) 
	except:
		print('Error getting position value from KB1')

	try:
		get_KB2(log=True, msg='KB2 position at the end of the shift on ' + now.strftime("%B-%d-%Y")) 
	except:
		print('Error getting position value from KB2')
	try:
		get_slits(slits=sl1k0, log=True, msg='SL1K0 position at the end of the shift on ' + now.strftime("%B-%d-%Y"))
		get_slits(slits=sl2k0, log=True, msg='SL2K0 position at the end of the shift on ' + now.strftime("%B-%d-%Y"))
		get_slits(slits=sl1k4, log=True, msg='SL1K4 position at the end of the shift on ' + now.strftime("%B-%d-%Y"))
		get_slits(slits=sl2k4, log=True, msg='SL2K4 position at the end of the shift on ' + now.strftime("%B-%d-%Y"))
	except:
		print('Error getting all the slits positions')
	print('Key component motor positions are logged in the current experiment logbook')

	#bringDownAll()
	#print('Bring all HVs down. Wait for 10s...')
	#time.sleep(10)
	#print('HVs all off.')
