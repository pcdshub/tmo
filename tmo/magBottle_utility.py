
'''
Voltage applying utility for Magnetic Bottle
author: Razib
Date created: 04/11/2021
Date last modified: 04/19/2021
'''

import numpy as np

from epics import caget, caput, cainfo
import time


def pv_ID(idString):
	'''Gives you pv strings for different voltage components. Please change the pv value in the magBottle_utility.py if any of the voltage pv changes during or before the experiment.
		Usage: pv_ID('anode') -- to get the pv for anode
		The allowed strings are: anode, mcpBack, mcpFront, mesh, back_lens, 			coil_bias'''
	if idString == 'anode': return 'TMO:MPOD:01:M2:C5:'
	elif idString == 'mcpBack': return 'TMO:MPOD:01:M2:C5:'
	elif idString == 'mcpFront': return 'TMO:MPOD:01:M2:C4:'
	elif idString == 'mesh': return 'TMO:MPOD:01:M2:C3:'
	elif idString == 'front_lens': return 'TMO:MPOD:01:M2:C0:'
	elif idString == 'back_lens': return 'TMO:MPOD:01:M2:C1:'
	elif idString == 'coil_bias': return 'TMO:MPOD:01:M2:C2:'
	#elif idString == 'pre-amp': return 'TMO:PWR:01:Outlet:6:'
	elif idString == 'test': return 'TMO:MPOD:01:M7:C2:'

def set_on_off(num):
	strings_to_on = ['anode', 'mcpBack', 'mcpFront', 'mesh', 'front_lens', 'back_lens', 'coil_bias']
	for st in strings_to_on:
		caput(pv_ID(st)+'Control:setOn', num)


def voltage(ret):
	MCP_mesh = ret
	MCP_front =  MCP_mesh + 300
	MCP_back = MCP_front + 1800
	Anode = MCP_back + 400
	print('MCP_mesh = ' + str(MCP_mesh) + ' V')
	print('MCP_front = ' + str(MCP_front) + ' V')
	print('MCP_back = ' + str(MCP_back) + ' V')
	print('Anode = ' + str(Anode) + ' V')

def voltage2(ret, mcpDiff=1700):
	MCP_mesh = ret
	MCP_front =  MCP_mesh + 300
	MCP_back = MCP_front + mcpDiff
	Anode = MCP_back + 400
	print('MCP_mesh = ' + str(MCP_mesh) + ' V')
	print('MCP_front = ' + str(MCP_front) + ' V')
	print('MCP_back = ' + str(MCP_back) + ' V')
	print('Anode = ' + str(Anode) + ' V')


def ramping_thru_MCP_front(setVoltage):
	'''Sets the voltages of MCP front, MCP back and anode at the same setvoltage.
		Usage: ramping_thru_MCP_front(300)
		this sets the voltages of front, back and anode to 300 V'''
	caput(pv_ID('mcpFront')+'VoltageSet', setVoltage)
	time.sleep(0.2)
	caput(pv_ID('mcpBack')+'VoltageSet', setVoltage)
	time.sleep(0.2)
	caput(pv_ID('anode')+'VoltageSet', setVoltage)
	time.sleep(0.2)

def ramping_thru_MCP_back(setVoltage):
	'''Sets the voltages of MCP back and anode at the same setvoltage.
		Usage: ramping_thru_MCP_back(300)
		this sets the voltages of back and anode to 300 V'''
	caput(pv_ID('mcpBack')+'VoltageSet', setVoltage)
	time.sleep(0.2)
	caput(pv_ID('anode')+'VoltageSet', setVoltage)
	time.sleep(0.2)


def ramping_anode_only(setVoltage):
	voltMCPBack = caget(pv_ID('mcpBack')+'VoltageMeasure')
	if (setVoltage < (np.int(voltMCPBack) + 402)):
		caput(pv_ID('anode')+'VoltageSet', setVoltage)
		time.sleep(0.2)
	else: print('Did not ramp anode to full voltage. The voltage difference between mcpBack and anode is too high')

def voltageSet(ret, mcpDiff=1700):

	'''This sets the voltages of coil, back lens, mesh, mcp front, mcp back and anode of the magnetic bottle spectrometer.
		Usage: voltageSet(retardation, mcpDiff)
			where retardation is voltage applied to coil, back lens and the mesh, mcpDiff is the voltage difference between the mcp front and the back. 	The anode is set to 400 V above the mcp back.
		Example: voltageSet(-100, 1000)'''
	set_on_off(1)
	if mcpDiff != 0:
		Coil_bias = np.max([0.5*ret, -200])
		Back_lens = ret
		Front_lens = 0   #ADDED 0N 03/20/2023			
		MCP_mesh = ret
		MCP_front =  MCP_mesh + 300
		MCP_back = MCP_front + mcpDiff
		Anode = MCP_back + 400
		

		caput(pv_ID('coil_bias')+'VoltageSet', Coil_bias)
		time.sleep(0.2)
		caput(pv_ID('back_lens')+'VoltageSet', Back_lens)
		time.sleep(0.2)
		caput(pv_ID('mesh')+'VoltageSet', MCP_mesh)
		time.sleep(0.2)
		
		ramping_thru_MCP_front(MCP_front)
		time.sleep(5.0)
		mcp_back_steps = np.linspace(MCP_front, MCP_back, 10)
		for volts in mcp_back_steps:
			ramping_thru_MCP_back(np.int(volts))
			time.sleep(2.0)
		voltMCPBack = caget(pv_ID('mcpBack')+'VoltageMeasure')
		while voltMCPBack < 0.98 * MCP_back: 
			time.sleep(1.0)
			voltMCPBack = caget(pv_ID('mcpBack')+'VoltageMeasure')
		time.sleep(1.0)
		ramping_anode_only(Anode)
	else: 
		print('Please do not use this to set mcpDiff = 0 in the future. \nUse either bringDownMCPs() or bringDownAll() instead. \nType "bringDownMCPs?" or "bringDownAll?" for help. \nFor now, using bringDownAll() by default for this command. :-)')
		bringDownAll()


def bringDownMCPs():
	'''Sets the voltages of MCP front, MCP back and anode at 0, 300 and 300 V respectively for safe tuning.
		Usage: bringDownMCPs()'''
	anode_pv = pv_ID('anode') + 'VoltageSet'
	mcpBack_pv = pv_ID('mcpBack') + 'VoltageSet'
	mcpFront_pv = pv_ID('mcpFront') + 'VoltageSet'
	caput(anode_pv, 300)
	time.sleep(0.2)
	caput(mcpBack_pv, 300)
	time.sleep(0.2)
	caput(mcpFront_pv, 0)

def bringDownAll():
	'''Sets the voltages of all relevant voltages for the magnetic bottle in the electron mode to zero.
		Usage: bringDownAll()'''
	anode_pv = pv_ID('anode') + 'VoltageSet'
	mcpBack_pv = pv_ID('mcpBack') + 'VoltageSet'
	mcpFront_pv = pv_ID('mcpFront') + 'VoltageSet'
	coil_pv = pv_ID('coil_bias') + 'VoltageSet'
	blens_pv = pv_ID('back_lens') + 'VoltageSet'
	flens_pv = pv_ID('front_lens') + 'VoltageSet'
	mesh_pv = pv_ID('mesh') + 'VoltageSet'

	caput(anode_pv, 0)
	time.sleep(0.2)
	caput(mcpBack_pv, 0)
	time.sleep(0.2)
	caput(mcpFront_pv, 0)
	time.sleep(0.2)
	caput(mesh_pv, 0)
	caput(blens_pv, 0)
	caput(flens_pv, 0)
	caput(coil_pv, 0)
	set_on_off(0)



def turn_on_off_preAmp(status):

	pre_amp_on = pv_ID('pre-amp')+'TurnOn'
	pre_amp_off = pv_ID('pre-amp')+'TurnOff'
	pre_amp_Status = pv_ID('pre-amp') + 'GetStatus'
	
	if status == 'ON':
		caput(pre_amp_on, 1)
		while caget(pre_amp_Status) != 2:
			time.sleep(0.2)
		print('The preamp is now on. Moving on to next step')
	if status == 'OFF':
		caput(pre_amp_off, 1)
		while caget(pre_amp_Status) != 1:
			time.sleep(0.2)
		print('The preamp is off. Moving on to next step')


def ramping_voltagePV(ID, setVoltage):
	caput(pv_ID(ID)+'VoltageSet', setVoltage)	
	voltRamp= caget(pv_ID(ID)+'VoltageMeasure')
	time.sleep(8.0)
	print('Ramping Done for ' + ID)

def ramping_voltagePV_2(ID, setVoltage):
	caput(pv_ID(ID)+'VoltageSet', setVoltage)	
	voltRamp = caget(pv_ID(ID)+'VoltageMeasure')
	
	diff = np.abs(voltRamp - setVoltage)
	while diff > 1:
		time.sleep(1.0)
		voltRamp = caget(pv_ID(ID)+'VoltageMeasure')
		diff = np.abs(voltRamp - setVoltage)
		
	print('Ramping done for '+ ID)

def ramping_voltage_incr(ID, setVoltages):
	voltRamp_init= caget(pv_ID(ID)+'VoltageMeasure')
	increment_Volt = np.linspace(voltRamp_init, setVoltages, 5)
	for setVoltage in increment_Volt:
		caput(pv_ID(ID)+'VoltageSet', setVoltage)	
		voltRamp= caget(pv_ID(ID)+'VoltageMeasure')
		time.sleep(6.0)
		voltRamp = caget(pv_ID(ID)+'VoltageMeasure')

def voltageRamp(ret):

	'''This sets the voltages of coil, back lens, mesh, mcp front, mcp back and anode of the magnetic bottle spectrometer.
		Usage: voltageSet(retardation, mcpDiff)
			where retardation is voltage applied to coil, back lens and the mesh, mcpDiff is the voltage difference between the mcp front and the back. 	The anode is set to 400 V above the mcp back.
		Example: voltageSet(-100, 1000)'''
	set_on_off(1)
	Coil_bias = np.max([0.5*ret, -200])
	Back_lens = ret
	MCP_mesh = ret


	mcpBack_Read = np.round(caget(pv_ID('mcpBack')+'VoltageSet'))
	mcpFront_Read = np.round(caget(pv_ID('mcpFront')+'VoltageSet'))
	ret_Read = np.int(np.round(caget(pv_ID('back_lens')+'VoltageSet')))
	
	mcpDiff = mcpBack_Read - mcpFront_Read #Step 0

	MCP_Back = mcpFront_Read + 1000
	ramping_voltagePV('mcpBack', MCP_Back)	#Step 1

	#turn_on_off_preAmp('OFF') #Step 2 and 3
	
	#Step 4, 5, 6
	if np.abs(ret) < np.abs(ret_Read):
		caput(pv_ID('back_lens')+'VoltageSet', Back_lens)
		time.sleep(0.2)
		caput(pv_ID('mesh')+'VoltageSet', MCP_mesh)
		time.sleep(0.2)
		caput(pv_ID('coil_bias')+'VoltageSet', Coil_bias)
		time.sleep(0.2)
		ramping_voltagePV('mcpFront', ret + 300)
		time.sleep(5.0)
		ramping_voltagePV('mcpBack', ret + 300 + 1000)
	else:
			
		ramping_voltagePV('mcpFront', ret + 300)
		time.sleep(0.5)
		ramping_voltagePV('mcpBack', ret + 300 + 1000)
		caput(pv_ID('coil_bias')+'VoltageSet', Coil_bias)
		time.sleep(0.2)
		caput(pv_ID('mesh')+'VoltageSet', MCP_mesh)
		time.sleep(0.2)
		caput(pv_ID('back_lens')+'VoltageSet', Back_lens)
		time.sleep(0.2)
	
	#turn_on_off_preAmp('ON') #Step 7
	print('Doing the final stage of ramping...')
	ramping_voltage_incr('mcpBack', ret + 300 + mcpDiff) #Step 8
	print('Ramping Done. Finished all steps.')

