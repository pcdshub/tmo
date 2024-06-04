#from tmo.vac_checkout import tmo_hvac, tmo_hvac_lamp, tmo_hvac_mrco, tmo_rough, tmo_lamp_rough, tmo_mrco_rough
from tmo.tmo_hvac import tmo_hvac, tmo_hvac_lamp, tmo_hvac_mrco, tmo_rough, tmo_lamp_rough, tmo_mrco_rough
from tmo.db import st3k4_pps_vgc_01, li3k4_ip1_vwc_01, pc5k4_vwc_01, mr3k4_kbo_vwc_01, mr3k4_kbo_vfs_01
import pandas as pd
import time
import matplotlib.pyplot as plt

granite_1_pump_list = ['pa1k4_ip1_ptm_03','pa1k4_ip1_ptm_02','pa1k4_ip1_ptm_01','im4k4_ppm_ptm_01','li1k4_ip1_ptm_01']
granite_2_pump_list = ['li3k4_ip1_out_ptm_01','tm1k4_ptm_01','im5k4_ppm_ptm_01']
granite_1_gauge_list = ['pa1k4_ip1_gcc_01','im4k4_ppm_gcc_01','li2k4_ip1_gfs_01','li2k4_ip1_gfs_02','li2k4_ip1_gfs_03']
granite_2_gauge_list = ['im5k4_ppm_gcc_01']#,'pa2k4_ip1_gcc_01']
pump_dict = {"MBES":['lamp_c2_ptm_01', 'lamp_c2_ptm_02', 'lamp_c2_ptm_03', 'lamp_c2_ptm_04', 'lamp_gj2_ptm_01', 'lamp_rough_ptm_01'], 
             "MRCO":['mrco_main_ptm_01', 'mrco_main_ptm_02', 'mrco_main_ptm_03', 'mrco_main_ptm_04', 'mrco_rough_ptm_01']}
gauge_dict = {"MBES":['lamp_c1_ghc_01', 'lamp_c2_gcc_01'],
              "MRCO":['mrco_main_ghc_01']}

_hv_gauge_protection_point = 5E-3 #interlock pressure for starting high vacuum gauges
_lv_gauge_setpoint = 5E-2 #interlock setpoint for roughing valves
_lv_gauge_pumpdown = 800 #interlock setpoint for rough valves to allow pump-down
_pump_backing_setpoint = 5E-2 #interlock setpoint for pump backing pressure
_pump_backing_setpoint_pumpdown = 2 #interlock setpoint for pump backing pressure during start up 

def verify_connected(tmo_hvac_ip, device_list):
    #for keys in dir(tmo_hvac):
    for keys in (granite_1_pump_list + granite_2_pump_list + granite_1_gauge_list + granite_2_gauge_list): 
        device = getattr(tmo_hvac, keys)
        if hasattr(device, 'connected'):
            if not bool(device.connected):
                print(device.name, ' not connected \n')
                return False
    for devices in device_list:
        device = getattr(tmo_hvac_ip, devices)
        if not bool(device.connected):
            print(device.name, ' not connected \n')
            return False
    return True 

def _are_pumps_running(hvac, pump_list, restart=False, at_speed=False):
    status=True
    for pump_name in pump_list:
        pump = getattr(hvac, pump_name)
        pump_state = pump.pump_state.get()
        if at_speed:
            pump_running = (pump_state == 2)    
        else:
            pump_running = (pump_state == 1) or (pump_state == 2)    
        if ( restart and ( (pump_state > 2) or (pump_state == 0) ) ):
            pump.switch_pump_on.put(1)
            print('Had to re-start ', pump.name)
            time.sleep(10)
            pump_state = pump.pump_state.get()
            pump_running = (pump_state == 1) or (pump_state == 2)
        status = status and (pump_running)
    return status

def _are_pumps_stopping(hvac, pump_list, stopped=False):
    status=True
    for pump_name in pump_list:
        pump = getattr(hvac,pump_name)
        pump_state = pump.pump_state.get()
        if stopped:
            status = status and (pump_state == 0)
        else:
            status = status and ( (pump_state == 0) or (pump_state == 4) )
    return status

def _start_pumps(hvac, pump_list):
    for pump_name in pump_list:
        pump = getattr(hvac, pump_name)
        pump.switch_pump_on.put(1)
        print("Starting ",pump.name)
        waiting = 0
        while not bool(pump.pump_state.get()): #(bool(pump.pump_accelerating.get()) or bool(pump.pump_at_speed.get())):
            time.sleep(1)
            waiting += 1
            if waiting>30:
                print(pump.name, ' failed to start in 30 sec!')
                return False
    return True

def _start_gauges(hvac, gauge_list):
    for gauge_name in gauge_list:
        gauge = getattr(hvac,gauge_name)
        if ( bool(gauge.interlock_ok.get()) and (not bool(gauge.high_voltage_on.get()) ) ):
            print('Starting ', gauge.name)
            gauge.high_voltage_on.put(1)
            time.sleep(1) 
    return True

def _stop_pumps(hvac, pump_list):
    for pump_name in pump_list:
        pump = getattr(hvac, pump_name)
        pump.switch_pump_on.put(0)
        print("Stopping ", pump.name)
    time.sleep(1)
    status = True
    for pump_name in pump_list:
        pump = getattr(hvac,pump_name)
        status = status and (not(pump.pump_state.get()==2))
    if not status:
        print('Some Pumps failed to stop.')
    return status  

def _stop_gauges(hvac, gauge_list):
    for gauge_name in gauge_list:
        gauge = getattr(hvac,gauge_name)
        gauge.high_voltage_on.put(0)
    return True

def _change_valve_setpoint(valve, setpoint):
    valve.at_vac_setpoint_ds.put(setpoint)
    valve.setpoint_hysterisis_ds.put(setpoint)
    valve.at_vac_setpoint_us.put(setpoint)
    valve.setpoint_hysterisis_us.put(setpoint)

def _change_gauge_setpoint(hvac, gauge_list, setpoint):
    for gauge_name in gauge_list:
        gauge = getattr(hvac, gauge_name)
        gauge.protection_setpoint.put(setpoint)

def _change_pump_backing_setpoint(hvac, pump_list, setpoint):
    for pump_name in pump_list:
        pump = getattr(hvac,pump_name)
        pump.backingpressure_sp.put(setpoint)
        pump.inletpressure_sp.put(setpoint)

def _open_roughing_valve(valve):
    valve.error_reset.put(1) #clear any error state
    time.sleep(1)
    valve.error_reset.put(0)
    time.sleep(1)
    if bool(valve.ext_ilk_ok.get()):
        valve.open_command.put(1)
        waiting = 0
        while bool(valve.valve_position.get()): #poll state
            time.sleep(0.5)
            waiting += 1
            if waiting > 20:
                print (valve.name,' failed to open in 10 seconds')
                return False
        return True
    else:
        print(valve.name, ' ext interlock has an issue')
        print('diff pressure Status: ', bool(valve.diff_press_ok.get())) 
        return False

def _open_vent_valve(valve, position):
    if bool(valve.interlock_ok.get()):
        valve.position_control.put(position)
        valve.state.put(3)
        valve.position_control.put(position)
#        valve.open_command.put(1)
        waiting = 0
        while not bool(valve.state.get()): #poll VFV state
            time.sleep(0.5)
            waiting += 1
            if waiting > 20:
                print (valve.name,' failed to open in 10 seconds')
                return False
#        print('Starting vent.....')
        return True
    else:
        print(valve.name, ' interlock has an issue.')
        return False
                
def pump_ip1(instrument='MBES', override_gauge=False):
    '''
    Pump down granite 1 and IP1 while granite 2 is at vacuum.

    Parameters:
    -----------
    instrument : str
                 name of instrument installed in IP1 (MBES, MRCO, CVMI)
    '''
    pump_list = pump_dict[instrument.upper()]
    gauge_list = gauge_dict[instrument.upper()]
    
    #define instrument
    if instrument.upper() == 'MBES':
        tmo_hvac_ip = tmo_hvac_lamp
        ip_vrc_01 = tmo_lamp_rough.lamp_rough_vrc_01
        ip_vrc_02 = tmo_lamp_rough.lamp_rough_vrc_02
        ip_vrc_03 = tmo_lamp_rough.lamp_rough_vrc_03
    elif instrument.upper() == 'MRCO':
        tmo_hvac_ip = tmo_hvac_mrco
        ip_vrc_01 = tmo_mrco_rough.mrco_rough_vrc_01
        ip_vrc_02 = tmo_mrco_rough.mrco_rough_vrc_02
        ip_vrc_03 = tmo_mrco_rough.mrco_rough_vrc_03
    else:
        print('Instrument not recognizied.')
        return

    #check that all required devices are connected
    if not verify_connected(tmo_hvac_ip, pump_list + gauge_list):
        print('Some devices not connected. Stopping pump down procedure. \n')
        return
    
    #check that ip1 is not at vac
    if (bool(tmo_hvac.tmo_ip1_main_gauge.gauge_at_vac.get()) and not override_gauge):
        print('IP1 main gauge says it is already at vacuum.\n Try again with override_gauge=True')
        return    

    #check that ip pumps are not running
    if _are_pumps_running(tmo_hvac_ip, pump_list):
        print('IP1 pumps are already running.')
        _start_gauges(tmo_hvac_ip, gauge_list)
        _start_gauges(tmo_hvac, granite_1_gauge_list)
        return
    
    #check that granite 1 pumps are not running
    if _are_pumps_running(tmo_hvac, granite_1_pump_list):
        print('Granite 1 pumps are already running.')
        _start_gauges(tmo_hvac_ip, gauge_list)
        _start_gauges(tmo_hvac, granite_1_gauge_list)
        return

    #check that fast valve is open
    if bool(mr3k4_kbo_vfs_01.valve_position.get()):
        print('MR3K4 fast valve is not open.')
        return
    
    #check that ip roughing valves are open
    status = bool(ip_vrc_01.valve_position.get()) or bool(ip_vrc_03.valve_position.get())
    if status: #returns True if closed 
        print('IP roughing valves are not open.')
        return

    # check granite 2 is at vacuum
    if not bool(tmo_hvac.im5k4_ppm_gcc_01.gauge_at_vac.get()):
        print('Granite 2 gauge (IM5K4) is not at vacuum.')
        if _are_pumps_running(tmo_hvac, granite_2_pump_list):
            print('Granite 2 pumps are running, starting gauge and continuing pump down.')
            _start_gauges(tmo_hvac, granite_2_gauge_list)
        else:
            print('Granite 2 pumps are not running. Use pump_all instead')
            return    

    #open roughing bypass valve
    #ip_vrc_02.open_command.put(1)
    if bool(ip_vrc_02.valve_position.get()):  #returns True if closed
        print('IP bypass valve not open')
        print('Opening bypass valve')
        if not _open_roughing_valve(ip_vrc_02):
            return

    #close vent valves
    print('Closing Vent Valves (Rough1:VFV:02, Rough2:VFV:01).......')
    tmo_rough.tmo_rough1_vfv_02.state.put(0)
    tmo_rough.tmo_rough2_vfv_01.state.put(0)
    
    #stop granite 2 pumps
    _stop_pumps(tmo_hvac, granite_2_pump_list)
    
    #close granite 2 roughing valve
    tmo_rough.tmo_rough1_vrc_02.open_command.put(0)
    time.sleep(5)
    if not bool(tmo_rough.tmo_rough1_vrc_02.valve_position.get()):
        print('Rough1:VRC:02 failed to close')
        return 

    #check granite 2 is off
    if not _are_pumps_stopping(tmo_hvac, granite_2_pump_list):
        print('Granite 2 pumps did not stop')
        return
    
    #change setpoints
    _change_valve_setpoint(tmo_rough.tmo_rough1_vrc_01, _lv_gauge_pumpdown)
    _change_valve_setpoint(tmo_rough.tmo_rough2_vrc_01, _lv_gauge_pumpdown)
    _change_gauge_setpoint(tmo_hvac, granite_1_gauge_list + granite_2_gauge_list, _hv_gauge_protection_point)
    _change_gauge_setpoint(tmo_hvac_ip, gauge_list, _hv_gauge_protection_point)
    _change_pump_backing_setpoint(tmo_hvac, granite_1_pump_list, 10*_pump_backing_setpoint_pumpdown)
    _change_pump_backing_setpoint(tmo_hvac_ip, pump_list, 10*_pump_backing_setpoint_pumpdown)

    #Open Rough1:VRC:01
    if not _open_roughing_valve(tmo_rough.tmo_rough1_vrc_01):
        return
    print('Starting Roughing....')
    _open_roughing_valve(tmo_rough.tmo_rough2_vrc_01)
    
    #wait for pumping
    waiting = 0
    print('TMO:Rough1:GPI:02: ', tmo_rough.tmo_rough1_gpi_02.pressure.get() )
    time.sleep(10)
    while (tmo_rough.tmo_rough1_gpi_02.pressure.get() > (_pump_backing_setpoint_pumpdown)):
        time.sleep(10)
        waiting += 1
        print('TMO:Rough1:GPI:02: ', tmo_rough.tmo_rough1_gpi_02.pressure.get())
        if waiting > 120:
            print('Failed to reach rough vacuum after 20 minutes...\n')
            tmo_rough.tmo_rough1_vrc_01.open_command.put(0)
            tmo_rough.tmo_rough2_vrc_01.open_command.put(0)
            print('Closed Rough1:VRC:01')
            return
    
    #start granite 1 pumps
    print('Starting Granite 1 and IP1 pumps....')
    _start_pumps(tmo_hvac, granite_1_pump_list)
    _start_pumps(tmo_hvac_ip, pump_list)

    #continue pumping
    waiting = 0
    while (tmo_rough.tmo_rough1_gpi_02.pressure.get() > 2e-2):
        time.sleep(10)
        waiting += 1
        print('TMO:Rough1:GPI:02: ', tmo_rough.tmo_rough1_gpi_02.pressure.get())
        
        #check if granite 1 gauges can start
        _start_gauges(tmo_hvac, granite_1_gauge_list)
        _start_gauges(tmo_hvac_ip, gauge_list)

        if waiting > 120:
            print('Failed to reach 2e-2 in Rough 1 after 20 minutes...\n')
            return  

    #check pumps are running
    if not(_are_pumps_running(tmo_hvac, granite_1_pump_list) and _are_pumps_running(tmo_hvac_ip, pump_list) ):
        print('IP1 and Granite 1 pumps are not starting....\n Stopping pumpdown')
        _stop_pumps(tmo_hvac, granite_1_pump_list)
        _stop_pumps(tmo_hvac_ip, pump_list)
        _change_pump_backing_setpoint(tmo_hvac, granite_1_pump_list, _pump_backing_setpoint)
        _change_pump_backing_setpoint(tmo_hvac_ip, pump_list, _pump_backing_setpoint)
        tmo_rough.tmo_rough1_vrc_01.open_command.put(0)
        tmo_rough.tmo_rough2_vrc_01.open_command.put(0)
        print('Closed Rough1:VRC:01')
        return
        
    #open rough1:VRC:02
    print('Opening Rough1:VRC:02.....')
    if not _open_roughing_valve(tmo_rough.tmo_rough1_vrc_02):
        print('Stopping pump down....')
        tmo_rough.tmo_rough1_vrc_01.open_command.put(0)
        tmo_rough.tmo_rough2_vrc_01.open_command.put(0)
        print('Closed Rough1:VRC:01')
        print('Check vacuum system and try pump_all(override_gauge=Ture)')
        return
   
    #Restart granite 2 pumps
    print('Restarting Granite 2 Pumps....')
    _start_pumps(tmo_hvac, granite_2_pump_list)
    _start_gauges(tmo_hvac, granite_2_gauge_list)
    print('All pumps started.....')

    #wait for pumps to reach speed and check if gauges can start
    waiting = 0 
    status = False
    while not status:
        time.sleep(10)

        #check if granite 1 gauges can start
        _start_gauges(tmo_hvac, granite_1_gauge_list)
        #check if ip1 gauges can start
        _start_gauges(tmo_hvac_ip, gauge_list)
        print("TMO:IP1:MainGauge:Pressure: ", tmo_hvac.tmo_ip1_main_gauge.pressure.get() )
        
        #check if pumps at speed
        status = True
        status = status and _are_pumps_running(tmo_hvac, granite_1_pump_list, restart=True, at_speed=True)
        status = status and _are_pumps_running(tmo_hvac, granite_2_pump_list, restart=True, at_speed=True)
        status = status and _are_pumps_running(tmo_hvac_ip, pump_list, restart=True, at_speed=True)

        waiting += 1
        if (waiting > 180):
            print('Some pumps failed to reach full speed after 30 minutes!')
            return 
    print('All pumps at speed....')

    #reset roughing setpoints
    while (tmo_rough.tmo_rough1_gpi_02.pressure.get() > (0.95*min([_lv_gauge_setpoint,_pump_backing_setpoint]))):
        time.sleep(10)
 
    _change_valve_setpoint(tmo_rough.tmo_rough1_vrc_01, _lv_gauge_setpoint)
    _change_valve_setpoint(tmo_rough.tmo_rough2_vrc_01, _lv_gauge_setpoint)
    _change_pump_backing_setpoint(tmo_hvac, granite_1_pump_list, _pump_backing_setpoint)
    _change_pump_backing_setpoint(tmo_hvac_ip, pump_list, _pump_backing_setpoint)

    return

def pump_granite_2(instrument="MBES", override_gauge=False):
    """
    Pump down Granite 2 while Granite 1 is at vacuum

    Parameters:
    -----------
    instrument : str 
                 name of instrument installed in IP1 (MBES, MRCO, CVMI)
    """
    pump_list = pump_dict[instrument.upper()]
    gauge_list = gauge_dict[instrument.upper()]
    
    if instrument.upper() == 'MBES':
        tmo_hvac_ip = tmo_hvac_lamp
    elif instrument.upper() == 'MRCO':
        tmo_hvac_ip = tmo_hvac_mrco

    if not verify_connected(tmo_hvac_ip, pump_list + gauge_list):
        print('Some devices not connected. Stopping pump down procedure. \n')
        return

    # check granite 2 not at vacuum
    if (bool(tmo_hvac.im5k4_ppm_gcc_01.gauge_at_vac.get()) and not override_gauge):
        print('Granite 2 gauge (IM5K4) says it is already at vacuum. \n Try again with override_gauge=True')
        return    
    
    #check that granite 2 pumps are not running
    if _are_pumps_running(tmo_hvac, granite_2_pump_list):
        print('Granite 2 pumps are already running.')
        _start_gauges(tmo_hvac, granite_2_gauge_list)
        return

    # check IP1 at vacuum
    if not bool(tmo_hvac.tmo_ip1_main_gauge.gauge_at_vac.get()):
        print('IP1 gauge is not at vacuum.')
        if (_are_pumps_running(tmo_hvac, granite_1_pump_list) and _are_pumps_running(tmo_hvac_ip, pump_list)):
            print('Granite 1 and IP1 pumps are running. Starting gauges and continuing pump down.')
            _start_gauges(tmo_hvac, granite_1_gauge_list)
        else:
            print('Granite 1 or IP1 pumps are not running, use pump_all instead.')
            return    

    #check that fast valve is open
    if bool(mr3k4_kbo_vfs_01.valve_position.get()):
        print('MR3K4 fast valve is not open.')
        return

    # close FEE gate valve
    print('Closing ST3K4.....')
    st3k4_pps_vgc_01.open_command.put(0)
    time.sleep(10)
    if not bool( st3k4_pps_vgc_01.valve_position.get() ):
        print('ST3K4 failed to close.')
        return

    # stop vent valve
    print('Closing Rough1:VFV:03....')
    tmo_rough.tmo_rough1_vfv_03.state.put(0)
    
    # stop pumps
    print('Stopping Granite 1 and IP1 pumps....')
    _stop_pumps(tmo_hvac, granite_1_pump_list)
    _stop_pumps(tmo_hvac_ip, pump_list)

    # Close Roughing valves
    print('Closing Rough1:VRC:01 and Rough2:VRC:01.....')
    tmo_rough.tmo_rough1_vrc_01.open_command.put(0)
    tmo_rough.tmo_rough2_vrc_01.open_command.put(0)
    time.sleep(5)
    if not (bool(tmo_rough.tmo_rough1_vrc_01.valve_position.get()) and bool(tmo_rough.tmo_rough2_vrc_01.valve_position.get())):
        print('Rough:VRC failed to close')
        return 

    #verify granite 1 and IP1 pumps are spinning down
    time.sleep(1)
    if not (_are_pumps_stopping(tmo_hvac_ip, pump_list) and _are_pumps_stopping(tmo_hvac, granite_1_pump_list)):
        print('Granite 1 pumps failed to stop.')
        return
    
    #change setpoints
    _change_valve_setpoint(tmo_rough.tmo_rough1_vrc_02, _lv_gauge_pumpdown)
    _change_gauge_setpoint(tmo_hvac, granite_1_gauge_list + granite_2_gauge_list, _hv_gauge_protection_point)
    _change_gauge_setpoint(tmo_hvac_ip, gauge_list, _hv_gauge_protection_point)
    _change_pump_backing_setpoint(tmo_hvac, granite_2_pump_list, 10*_pump_backing_setpoint_pumpdown)

    # Open VRC 02
    if not _open_roughing_valve(tmo_rough.tmo_rough1_vrc_02):
        return
    print('Starting Roughing....')
    
    #wait for pump down
    waiting = 0
    print('TMO:Rough1:GPI:03: ', tmo_rough.tmo_rough1_gpi_03.pressure.get())
    time.sleep(1)
    while (tmo_rough.tmo_rough1_gpi_03.pressure.get() > (_pump_backing_setpoint_pumpdown)):
        time.sleep(10)
        waiting += 1
        print('TMO:Rough1:GPI:03: ', tmo_rough.tmo_rough1_gpi_03.pressure.get() )
        if waiting > 120:
            print('Failed to reach rough vacuum after 20 minutes...\n')
            tmo_rough.tmo_rough1_vrc_02.open_command.put(0)
            print('Closed Rough1:VRC:02')
            return

    #start granite 2 pumps
    print('Starting Granite 2 pumps....')
    _start_pumps(tmo_hvac, granite_2_pump_list)
    
    #wait for pump down
    waiting = 0
    while (tmo_rough.tmo_rough1_gpi_03.pressure.get() > 1e-2):
        time.sleep(10)
        waiting += 1
        print('TMO:Rough1:GPI:03: ', tmo_rough.tmo_rough1_gpi_03.pressure.get())
        
        #check if granite 2 gauges can start
        _start_gauges(tmo_hvac, granite_2_gauge_list)

        if waiting > 120:
            print('Failed to reach 1e-2 after 20 minutes...\n')
            return

    if not _are_pumps_running(tmo_hvac, granite_2_pump_list):
        print('Granite 2 pumps stopped running.....\n Stopping pump down')
        _stop_pumps(tmo_hvac, granite_2_pump_list)
        _change_pump_backing_setpoint(tmo_hvac, granite_2_pump_list, _pump_backing_setpoint)
        tmo_rough.tmo_rough1_vrc_02.open_command.put(0)
        print('Closed Rough1:VRC:02')
        return
    
    #open Rough1:VRC:01
    print('Opening Rough1:VRC:01.....')
    if not _open_roughing_valve(tmo_rough.tmo_rough1_vrc_01):
        print('Stopping pump down....')
        tmo_rough.tmo_rough1_vrc_02.open_command.put(0)
        print('Closed Rough1:VRC:02')
        print('Check vacuum system and try pump_all(override_gauge=Ture)')
        return
 
    #Restart IP1 and Granite 1 Pumps
    _start_pumps(tmo_hvac, granite_1_pump_list)
    _start_pumps(tmo_hvac_ip, pump_list)
    print('All pumps started.....')

    #Start high vacuum gauges
    _start_gauges(tmo_hvac, granite_1_gauge_list)
    _start_gauges(tmo_hvac_ip, gauge_list)


    #wait for pumps to reach speed
    waiting = 0 
    status = False
    while not status:
        time.sleep(10)
        #Start high vacuum gauges
        _start_gauges(tmo_hvac, granite_2_gauge_list)
        print("IM5K4:PPM:GCC:01:Pressure: ", tmo_hvac.im5k4_ppm_gcc_01.pressure.get() )

        #check if pumps at speed
        status = True
        status = status and _are_pumps_running(tmo_hvac, granite_1_pump_list, restart=True, at_speed=True)
        status = status and _are_pumps_running(tmo_hvac, granite_2_pump_list, restart=True, at_speed=True)
        status = status and _are_pumps_running(tmo_hvac_ip, pump_list, restart=True, at_speed=True)

        waiting += 1
        if (waiting > 180):
            print('Some pumps failed to reach full speed after 30 minutes!')
            return 

    print('All pumps at speed....')

    #reset setpoints
    while (tmo_rough.tmo_rough1_gpi_03.pressure.get() > (0.95*min([_lv_gauge_setpoint,_pump_backing_setpoint]))):
        time.sleep(10)
    _change_valve_setpoint(tmo_rough.tmo_rough1_vrc_02, _lv_gauge_setpoint)
    _change_pump_backing_setpoint(tmo_hvac, granite_2_pump_list, _pump_backing_setpoint)

    return

def pump_all(instrument="MBES", override_gauge=False):
    """
    Pump down the entire beamline (both Granite 1, IP1, and Granite 2)
    
    Parameters:
    -----------
    instrument : str 
            name of instrument installed in IP1 (MBES, MRCO, CVMI)
    override_gauge : boolean, optional
            ignore if some gauges are already at vacuum. 
    """
    #pump Granite 2, then pump Ip1 and Granite 1

    pump_list = pump_dict[instrument.upper()]
    gauge_list = gauge_dict[instrument.upper()]
    
    #define instrument
    if instrument.upper() == 'MBES':
        tmo_hvac_ip = tmo_hvac_lamp
        ip_vrc_01 = tmo_lamp_rough.lamp_rough_vrc_01
        ip_vrc_02 = tmo_lamp_rough.lamp_rough_vrc_02
        ip_vrc_03 = tmo_lamp_rough.lamp_rough_vrc_03
    elif instrument.upper() == 'MRCO':
        tmo_hvac_ip = tmo_hvac_mrco
        ip_vrc_01 = tmo_mrco_rough.mrco_rough_vrc_01
        ip_vrc_02 = tmo_mrco_rough.mrco_rough_vrc_02
        ip_vrc_03 = tmo_mrco_rough.mrco_rough_vrc_03
    else:
        print('Instrument not recognizied.')
        return

    if not verify_connected(tmo_hvac_ip, pump_list + gauge_list):
        print('Some devices not connected. Stopping pump down procedure. \n')
        return

    # check granite 2 not at vacuum
    if ( bool(tmo_hvac.im5k4_ppm_gcc_01.gauge_at_vac.get()) and not override_gauge ):
        print('Granite 2 gauge (IM5K4) says it is already at vacuum.')
        return    
    
    #check that ip1 is not at vac
    if ( bool(tmo_hvac.tmo_ip1_main_gauge.gauge_at_vac.get()) and not override_gauge ):
        print('IP1 gauge says it is already at vacuum.')
        return    

    #check that granite 2 pumps are not running
    if _are_pumps_running(tmo_hvac, granite_2_pump_list):
        print('Granite 2 pumps are already running.')
        _start_gauges(tmo_hvac, granite_2_gauge_list)
        return
    
    #check that ip pumps are not running
    if _are_pumps_running(tmo_hvac_ip, pump_list):
        print('IP1 pumps are already running.')
        _start_gauges(tmo_hvac_ip, gauge_list)
        return
    
    #check that granite 1 pumps are not running
    if _are_pumps_running(tmo_hvac, granite_1_pump_list):
        print('Granite 1 pumps are already running.')
        _start_gauges(tmo_hvac, granite_1_gauge_list)
        return

    #check that fast valve is open
    if bool(mr3k4_kbo_vfs_01.valve_position.get()):
        print('Fast valve is not open.')
        return

    #check that ip roughing valves are open
    status = bool(ip_vrc_01.valve_position.get()) or bool(ip_vrc_03.valve_position.get())
    if status: #returns True if closed 
        print('IP roughing valves are not open.')
        return

    #open roughing bypass valve
    if bool(ip_vrc_02.valve_position.get()):  #returns True if closed
        print('IP bypass valve not open')
        print('Opening IP bypass valve. Try running script again.')
        return

    # close FEE gate valve
    print('Closing ST3K4.....')
    st3k4_pps_vgc_01.open_command.put(0)
    time.sleep(10)
    if not bool( st3k4_pps_vgc_01.valve_position.get() ):
        print('ST3K4 failed to close.')
        return

    #close vent valves
    print('Closing vent valves.....')
    tmo_rough.tmo_rough1_vfv_02.state.put(0)
    tmo_rough.tmo_rough1_vfv_03.state.put(0)
    tmo_rough.tmo_rough2_vfv_01.state.put(0)
    
    # Close Roughing valves
    print('Closing Rough1:VRC:01 and Rough2:VRC:01.....')
    tmo_rough.tmo_rough1_vrc_01.open_command.put(0)
    tmo_rough.tmo_rough2_vrc_01.open_command.put(0)
    time.sleep(10)
    if not (bool(tmo_rough.tmo_rough1_vrc_01.valve_position.get()) and bool(tmo_rough.tmo_rough2_vrc_01.valve_position.get())):
        print('Rough:VRC failed to close')
        return 
    
    #change setpoints
    _change_valve_setpoint(tmo_rough.tmo_rough1_vrc_02, _lv_gauge_pumpdown)
    _change_gauge_setpoint(tmo_hvac, granite_1_gauge_list + granite_2_gauge_list, _hv_gauge_protection_point)
    _change_pump_backing_setpoint(tmo_hvac, granite_1_pump_list, 10*_pump_backing_setpoint_pumpdown)
    _change_pump_backing_setpoint(tmo_hvac, granite_2_pump_list, 10*_pump_backing_setpoint_pumpdown)
    _change_pump_backing_setpoint(tmo_hvac_ip, pump_list, 10*_pump_backing_setpoint_pumpdown)
     
    # Open VRC 02
    if not _open_roughing_valve(tmo_rough.tmo_rough1_vrc_02):
        return
    
    #wait for pump down
    waiting = 0
    print('TMO:Rough1:GPI:03: ', tmo_rough.tmo_rough1_gpi_03.pressure.get())
    time.sleep(1)
    while (tmo_rough.tmo_rough1_gpi_03.pressure.get() > (_pump_backing_setpoint_pumpdown)):
        time.sleep(10)
        waiting += 1
        print('TMO:Rough1:GPI:03: ', tmo_rough.tmo_rough1_gpi_03.pressure.get() )
        if waiting > 120:
            print('Failed to reach rough vacuum after 20 minutes...\n')
            tmo_rough.tmo_rough1_vrc_02.open_command.put(0)
            print('Closed Rough1:VRC:02')
            return

    #start granite 2 pumps
    print('Starting Granite 2 pumps....')
    _start_pumps(tmo_hvac, granite_2_pump_list)
  
    #wait for pumps to spin up
    print('Waiting for Granite 2 pumps to spin up....')
    status = False
    while not status:
        status = _are_pumps_running(tmo_hvac, granite_2_pump_list, restart=True, at_speed=True)    
        time.sleep(10)
        #check if granite 2 gauges can start
        _start_gauges(tmo_hvac, granite_2_gauge_list)
        print('IM5K4 Pressure: ',tmo_hvac.im5k4_ppm_gcc_01.pressure.get())

    print('Granite 2 pumps spun up. Stopping to pump granite 1')
    _stop_pumps(tmo_hvac, granite_2_pump_list)
    
    #close granite 2 roughing valve
    print('Closing Rough1:VRC:02.....')
    tmo_rough.tmo_rough1_vrc_02.open_command.put(0)
    time.sleep(10)
    if not bool(tmo_rough.tmo_rough1_vrc_02.valve_position.get()):
        print('Rough1:VRC:02 failed to close')
        return 

    #check granite 2 is off
    if not _are_pumps_stopping(tmo_hvac, granite_2_pump_list):
        print('Granite 2 pumps did not stop')
        return

    #change setpoints
    _change_valve_setpoint(tmo_rough.tmo_rough1_vrc_01, _lv_gauge_pumpdown)
    _change_valve_setpoint(tmo_rough.tmo_rough2_vrc_01, _lv_gauge_pumpdown)

    #Open Rough1:VRC:01
    if not _open_roughing_valve(tmo_rough.tmo_rough1_vrc_01):
        return
    _open_roughing_valve(tmo_rough.tmo_rough2_vrc_01)

    #wait for pumping
    waiting = 0
    print('TMO:Rough1:GPI:02: ', tmo_rough.tmo_rough1_gpi_02.pressure.get() )
    while (tmo_rough.tmo_rough1_gpi_02.pressure.get() > (_pump_backing_setpoint_pumpdown)):
        time.sleep(10)
        #check if granite 2 gauges can start
        _start_gauges(tmo_hvac, granite_2_gauge_list)
        waiting += 1
        print('TMO:Rough1:GPI:02: ', tmo_rough.tmo_rough1_gpi_02.pressure.get())
        if waiting > 120:
            print('Granite 1 failed to reach rough vacuum after 20 minutes...\n')
            tmo_rough.tmo_rough1_vrc_01.open_command.put(0)
            tmo_rough.tmo_rough2_vrc_01.open_command.put(0)
            print('Closed Rough1:VRC:01')
            return
    
    #start granite 1 pumps
    print('Starting Granite 1 and IP1 pumps....')
    _start_pumps(tmo_hvac, granite_1_pump_list)
    _start_pumps(tmo_hvac_ip, pump_list)

    #continue pumping
    waiting = 0
    while (tmo_rough.tmo_rough1_gpi_02.pressure.get() > 2e-2):
        time.sleep(10)
        waiting += 1
        print('TMO:Rough1:GPI:02: ', tmo_rough.tmo_rough1_gpi_02.pressure.get())
        
        #check if granite 1 gauges can start
        _start_gauges(tmo_hvac, granite_1_gauge_list)        
        #check ip1 gauges
        _start_gauges(tmo_hvac_ip, gauge_list)
        #check if granite 2 gauges can start
        _start_gauges(tmo_hvac, granite_2_gauge_list)

        if waiting > 120:
            print('Failed to reach 2e-2 in Rough 1 after 20 minutes...\n')
            return
    
    #open rough1:VRC:02
    print('Opening Rough1:VRC:02.....')
    if not _open_roughing_valve(tmo_rough.tmo_rough1_vrc_02):
        print('Stopping pump down....')
        tmo_rough.tmo_rough1_vrc_01.open_command.put(0)
        tmo_rough.tmo_rough2_vrc_01.open_command.put(0)
        print('Closed Rough1:VRC:01')
        print('Check vacuum system and try pump_all(override_gauge=Ture)')
        return
    
    #Restart granite 2 pumps
    print('Restarting Granite 2 Pumps....')
    _start_pumps(tmo_hvac, granite_2_pump_list)
    _start_gauges(tmo_hvac, granite_2_gauge_list)
    
    print('All pumps started.....')

    #wait for pumps to reach speed
    waiting = 0 
    status = False
    while not status:
        time.sleep(10)

        _start_gauges(tmo_hvac, granite_1_gauge_list)
        _start_gauges(tmo_hvac, granite_2_gauge_list)
        _start_gauges(tmo_hvac_ip, gauge_list)
        print("TMO:IP1:MainGauge:Pressure: ", tmo_hvac.tmo_ip1_main_gauge.pressure.get() )
        
        #check if pumps are at speed.
        status = True
        status = status and _are_pumps_running(tmo_hvac, granite_2_pump_list, restart=True, at_speed=True)
        status = status and _are_pumps_running(tmo_hvac, granite_1_pump_list, restart=True, at_speed=True)
        status = status and _are_pumps_running(tmo_hvac_ip, pump_list, restart=True, at_speed=True)
        
        waiting += 1
        if (waiting > 180):
            print('Some pumps failed to reach full speed after 30 minutes!')
            return 
    print('All pumps at speed....')

    #reset setpoints
    while (tmo_rough.tmo_rough1_gpi_02.pressure.get() > (0.95*min([_lv_gauge_setpoint,_pump_backing_setpoint]))):
        time.sleep(10)
    _change_valve_setpoint(tmo_rough.tmo_rough1_vrc_01, _lv_gauge_setpoint)
    _change_valve_setpoint(tmo_rough.tmo_rough2_vrc_01, _lv_gauge_setpoint)
    _change_pump_backing_setpoint(tmo_hvac, granite_1_pump_list, _pump_backing_setpoint)
    _change_pump_backing_setpoint(tmo_hvac_ip, pump_list, _pump_backing_setpoint)

    while (tmo_rough.tmo_rough1_gpi_03.pressure.get() > (0.95*min([_lv_gauge_setpoint,_pump_backing_setpoint]))):
        time.sleep(10)    
    _change_valve_setpoint(tmo_rough.tmo_rough1_vrc_02, _lv_gauge_setpoint)
    _change_pump_backing_setpoint(tmo_hvac, granite_2_pump_list, _pump_backing_setpoint)

    return

def vent_ip1(instrument="MBES"):
    """
    Vent IP1 and Granite 1
    
    Parameters:
    -----------
    instrument : str 
                 name of instrument installed in IP1 (MBES, MRCO, CVMI)
    """    
    pump_list = pump_dict[instrument.upper()]
    gauge_list = gauge_dict[instrument.upper()]
    
    #define instrument
    if instrument.upper() == 'MBES':
        tmo_hvac_ip = tmo_hvac_lamp
        ip_vrc_01 = tmo_lamp_rough.lamp_rough_vrc_01
        ip_vrc_02 = tmo_lamp_rough.lamp_rough_vrc_02
        ip_vrc_03 = tmo_lamp_rough.lamp_rough_vrc_03
    elif instrument.upper() == 'MRCO':
        tmo_hvac_ip = tmo_hvac_mrco
        ip_vrc_01 = tmo_mrco_rough.mrco_rough_vrc_01
        ip_vrc_02 = tmo_mrco_rough.mrco_rough_vrc_02
        ip_vrc_03 = tmo_mrco_rough.mrco_rough_vrc_03
    else:
        print('Instrument not recognizied.')
        return

    #check that all required devices are connected
    if not verify_connected(tmo_hvac_ip, pump_list + gauge_list):
        print('Some devices not connected. Stopping vent procedure. \n')
        return    

    #Check if already vented
    if (tmo_rough.pa1k4_ip1_gpi_01.pressure.get() > 500):
        print('Granite 1 already vented.')
        return

    #Close all gate valves in beamline
    print('Closing gate valves......')
    tmo_hvac.close_valves()
    time.sleep(10)

    #Check important valves are closed
    if not ( bool(mr3k4_kbo_vwc_01.valve_position.get()) and bool(li3k4_ip1_vwc_01.valve_position.get()) and bool(st3k4_pps_vgc_01.valve_position.get()) ):
        print('Gate valves did not close')
        return 
    
    #Open Fast Valve
#    mr3k4_kbo_vfs_01.valve_position.request_open.put(1)
#    time.sleep(2)
    if bool(mr3k4_kbo_vfs_01.valve_position.get()):
        print('MR3K4 Fast valve is not open.')
        return

    #verify by-pass valve is open
    if bool(ip_vrc_02.valve_position.get()):  #returns True if closed
        print('IP bypass valve not open')
        print('opening bypass valve')
        if not _open_roughing_valve(ip_vrc_02):
            return

    #Stop Granite 1 Pumps
    print('Stopping Granite 1 pumps.....')
    _stop_pumps(tmo_hvac, granite_1_pump_list)
    
    #Stop IP1 Pumps
    print('Stopping IP1 pumps.....')
    _stop_pumps(tmo_hvac_ip, pump_list)
 
    #close Roughing Line
    print('Closing roughing lines for IP1 and Granite 1.....')
    tmo_rough.tmo_rough1_vrc_01.open_command.put(0)
    tmo_rough.tmo_rough2_vrc_01.open_command.put(0)
    time.sleep(1)    

    #Check pumps have stopped
    status = _are_pumps_stopping(tmo_hvac, granite_1_pump_list)
    status = status and _are_pumps_stopping(tmo_hvac_ip, pump_list)
    if not status:
        print('Some Pumps failed to stop.')
        return  

    time.sleep(1)
    #check roughing closed
    if not (bool(tmo_rough.tmo_rough1_vrc_01.valve_position.get()) and bool(tmo_rough.tmo_rough2_vrc_01.valve_position.get()) ):
        print('Roughing valves failed to close')
        return

    #Stop high vacuum gauges
    print('Stopping HV gauges.....')
    _stop_gauges(tmo_hvac, granite_1_gauge_list)
    _stop_gauges(tmo_hvac_ip, gauge_list)

    time.sleep(60)
    #Open VFV Vent Valve
    print('Opening Vent Valve Rough1:VFV:02....')
    vent_valve_opening = 7
    _open_vent_valve(tmo_rough.tmo_rough1_vfv_02, vent_valve_opening)    
    
    timer = 0
    #wait for pumps to stop
    status = False
    while not status:
        time.sleep(10)
        timer+=1
        
        if (timer%12 == 0): #Increase opening every 2 minutes
            vent_valve_opening+=1
            print('Increasing Rough1:VFV:02 Opening')
            _open_vent_valve(tmo_rough.tmo_rough1_vfv_02, vent_valve_opening)
        
        status = True
        for pump_name in granite_1_pump_list:
            pump = getattr(tmo_hvac, pump_name)
            status = status and (pump.pump_speed.get() < 10)
        for pump_name in pump_list:
            pump = getattr(tmo_hvac_ip, pump_name)
            status = status and (pump.pump_speed.get() < 10)
    
    print('IP1 and Granite 1 pumps have stopped.')                        
    print('Opening Vent Valve Rough1:VFV:02 to 100%')
    _open_vent_valve(tmo_rough.tmo_rough1_vfv_02, 100)

    #wait for venting
    while (tmo_rough.tmo_rough1_gpi_02.pressure.get() < 760):
        time.sleep(10)
    tmo_rough.tmo_rough1_vfv_02.state.put(0)
    print('IP1 and Granite 1 vented.')
    return

def vent_granite_2(instrument="MBES"):
    """
    Vent Granite 2 only
    
    Parameters:
    -----------
    instrument : str 
                 name of instrument installed in IP1 (MBES, MRCO, CVMI)
    """
    pump_list = pump_dict[instrument.upper()]
    gauge_list = gauge_dict[instrument.upper()]
    
    #define instrument
    if instrument.upper() == 'MBES':
        tmo_hvac_ip = tmo_hvac_lamp
        ip_vrc_01 = tmo_lamp_rough.lamp_rough_vrc_01
        ip_vrc_02 = tmo_lamp_rough.lamp_rough_vrc_02
        ip_vrc_03 = tmo_lamp_rough.lamp_rough_vrc_03
    elif instrument.upper() == 'MRCO':
        tmo_hvac_ip = tmo_hvac_mrco
        ip_vrc_01 = tmo_lamp_rough.mrco_rough_vrc_01
        ip_vrc_02 = tmo_lamp_rough.mrco_rough_vrc_02
        ip_vrc_03 = tmo_lamp_rough.mrco_rough_vrc_03
    else:
        print('Instrument not recognizied.')
        return

    #check that all required devices are connected
    if not verify_connected(tmo_hvac_ip, pump_list + gauge_list):
        print('Some devices not connected. Stopping vent procedure. \n')
        return    

    #Check if already vented
    if (tmo_rough.tmo_rough1_gpi_03.pressure.get() > 500):
        print('Granite 2 already vented.')
        return

    #Close all gate valves in beamline
    print('Closing gate valves......')
    tmo_hvac.close_valves()
    time.sleep(1)

    #Check important valves are closed
    if not ( bool(li3k4_ip1_vwc_01.valve_position.get()) and bool(pc5k4_vwc_01.valve_position.get()) ):
        print('Gate valves did not close')
        return 
    
    #Stop Granite 2 Pumps
    print('Stopping Granite 2 pumps.....')
    _stop_pumps(tmo_hvac, granite_2_pump_list)
 
    #close Roughing Line
    print('Closing roughing lines for Granite 2.....')
    tmo_rough.tmo_rough1_vrc_02.open_command.put(0)
    time.sleep(1) 

    #Check pumps have stopped   
    if not _are_pumps_stopping(tmo_hvac, granite_2_pump_list):
        print('Some Pumps failed to stop.')
        return  

    time.sleep(1)
    #check roughing closed
    if not bool(tmo_rough.tmo_rough1_vrc_02.valve_position.get()):
        print('Roughing valve failed to close')
        return

    #Stop high vacuum gauges
    print('Stopping HV gauges.....')
    _stop_gauges(tmo_hvac, granite_2_gauge_list)

    time.sleep(60)
    #Open VFV Vent Valve
    print('Opening Vent Valve Rough1:VFV:03....')
    vent_valve_opening = 7
    _open_vent_valve(tmo_rough.tmo_rough1_vfv_03, vent_valve_opening)    
    
    timer = 0
    #wait for pumps to stop
    status = False
    while not status:
        time.sleep(10)
        timer+=1
        
        if (timer%12 == 0): #Increase opening every 2 minutes
            vent_valve_opening+=1
            print('Increasing Rough1:VFV:03 Opening')
            _open_vent_valve(tmo_rough.tmo_rough1_vfv_03, vent_valve_opening)

        status = True
        for pump_name in granite_2_pump_list:
            pump = getattr(tmo_hvac, pump_name)
            status = status and (pump.pump_speed.get() < 10)
        
    print('Granite 2 pumps have stopped.')
    
    print('Opening Vent Valve Rough1:VFV:03 to 100%')
    _open_vent_valve(tmo_rough.tmo_rough1_vfv_03, 100)                        

    #wait for venting
    while (tmo_rough.tmo_rough1_gpi_03.pressure.get() < 760):
        time.sleep(10)
    tmo_rough.tmo_rough1_vfv_03.state.put(0)
    print('Granite 2 vented.')
    return
