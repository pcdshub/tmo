from ophyd.signal import EpicsSignal, EpicsSignalRO
import logging
import time
logger = logging.getLogger(__name__)

alarm_dict = {
    'MR1K4:SOMS:': {
                    'COATING:STATE:GET_RBV':{'Thresholds':None, 'Filter':'SIOC:SYS0:ML00:AO628 <= 420'},
        'COATING:STATE:PMPS:ARB:ENABLE_RBV':{'Thresholds':None, 'Filter': None},
                                'FWM:1_RBV':{'Thresholds':{'LOLO':1.5,'LOW':1.7,'HIGH':3.0,'HIHI':4.0}, 'Filter':None},
                                'FWM:2_RBV':{'Thresholds':{'LOLO':1.5,'LOW':1.7,'HIGH':3.0,'HIHI':4.0}, 'Filter':None},
                             'GANTRY_X_RBV':{'Thresholds':{'LOLO':-20,'LOW':-10,'HIGH':10,'HIHI':20}, 'Filter':None},
                             'GANTRY_Y_RBV':{'Thresholds':{'LOLO':-20,'LOW':-10,'HIGH':10,'HIHI':20}, 'Filter':None},
                          'GCC:1:PRESS_RBV':{'Thresholds':{'LOLO':0,'LOW':0,'HIGH':5E-9,'HIHI':1E-8}, 'Filter':None},
                         'PIP:01:PRESS_RBV':{'Thresholds':{'LOLO':0,'LOW':0,'HIGH':5E-9,'HIHI':1E-8}, 'Filter':None},
                         'PRSM:1_RBV':{'Thresholds':{'LOLO':0.00,'LOW':0.1,'HIGH':4.0,'HIHI':5.0}, 'Filter':None}
        },
    'MR2K4:KBO:': {
                    'COATING:STATE:GET_RBV':{'Thresholds':None, 'Filter':'SIOC:SYS0:ML00:AO628 <= 420'},
        'COATING:STATE:PMPS:ARB:ENABLE_RBV':{'Thresholds':None, 'Filter':None},
                                'FWM:1_RBV':{'Thresholds':{'LOLO':1.5,'LOW':1.7,'HIGH':2.3,'HIHI':3.0}, 'Filter':None},
                          'GCC:01:PRESS_RBV':{'Thresholds':{'LOLO':0,'LOW':0,'HIGH':5E-9,'HIHI':1E-8}, 'Filter':None},
                         'PIP:01:PRESS_RBV':{'Thresholds':{'LOLO':0,'LOW':0,'HIGH':5E-9,'HIHI':1E-8}, 'Filter':None},
                               'PRSM:1_RBV':{'Thresholds':{'LOLO':0.00,'LOW':0.1,'HIGH':3,'HIHI':4}, 'Filter':None},
                      'RTD:CHIN:L:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':80,'HIHI':100}, 'Filter':None},
                      'RTD:CHIN:R:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':80,'HIHI':100}, 'Filter':None},
        },
    'MR3K4:KBO:': {
                    'COATING:STATE:GET_RBV':{'Thresholds':None, 'Filter':'SIOC:SYS0:ML00:AO628 <= 420'},
        'COATING:STATE:PMPS:ARB:ENABLE_RBV':{'Thresholds':None, 'Filter':None},
                                'FWM:1_RBV':{'Thresholds':{'LOLO':1.5,'LOW':1.7,'HIGH':2.3,'HIHI':3.0}, 'Filter':None},
                      'RTD:CHIN:L:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':80,'HIHI':100}, 'Filter':None},
                      'RTD:CHIN:R:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':80,'HIHI':100}, 'Filter':None},        
        },
    'IM1K4:XTES:': {
            'MMS:STATE:PMPS:ARB:ENABLE_RBV':{'Thresholds':None, 'Filter':None},        
        },
    'IM2K4:PPM:':{
            'MMS:STATE:PMPS:ARB:ENABLE_RBV':{'Thresholds':None, 'Filter':None},
                         'SPM:STC:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
                         'YAG:STC:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
                              'FWM:VAL_RBV':{'Thresholds':{'LOLO':1.7,'LOW':1.9,'HIGH':3.0,'HIHI':4.0}, 'Filter':None}, 
        },
    'IM3K4:PPM:':{
            'MMS:STATE:PMPS:ARB:ENABLE_RBV':{'Thresholds':None, 'Filter':None},
                         'SPM:STC:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
                         'YAG:STC:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
        },
    'IM4K4:PPM:':{
            'MMS:STATE:PMPS:ARB:ENABLE_RBV':{'Thresholds':None, 'Filter':None},
                         'SPM:STC:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
                         'YAG:STC:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
                              'FWM:VAL_RBV':{'Thresholds':{'LOLO':1.7,'LOW':1.9,'HIGH':3.0,'HIHI':4.0}, 'Filter':None}, 
        },
    'IM5K4:PPM:':{
            'MMS:STATE:PMPS:ARB:ENABLE_RBV':{'Thresholds':None, 'Filter':None},
                         'SPM:STC:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
                         'YAG:STC:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
                              'FWM:VAL_RBV':{'Thresholds':{'LOLO':3.5,'LOW':3.7,'HIGH':5.0,'HIHI':6.0}, 'Filter':None}, 
                       'CAM:TSS:SyncStatus':{'Thresholds':None, 'Filter':'IM5K4:PPM:CAM:TriggerMode_RBV!=0'}, 
                     'CAM:TSS:DiffVsIntAvg':{'Thresholds':{'LOLO':-1.5,'LOW':-1.5,'HIGH':1.5,'HIHI':1.5}, 'Filter':None}, 
        },
    'IM6K4:PPM:':{
            'MMS:STATE:PMPS:ARB:ENABLE_RBV':{'Thresholds':None, 'Filter':None},
                         'SPM:STC:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
                         'YAG:STC:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
                              'FWM:VAL_RBV':{'Thresholds':{'LOLO':3.0,'LOW':3.2,'HIGH':5.0,'HIHI':6.0}, 'Filter':None}, 
                       'CAM:TSS:SyncStatus':{'Thresholds':None, 'Filter':'IM6K4:PPM:CAM:TriggerMode_RBV!=0'}, 
                     'CAM:TSS:DiffVsIntAvg':{'Thresholds':{'LOLO':-1.5,'LOW':-1.5,'HIGH':1.5,'HIHI':1.5}, 'Filter':None},  
        },
    'PF1K4:WFS:':{
            'MMS:STATE:PMPS:ARB:ENABLE_RBV':{'Thresholds':None, 'Filter':None},
                          'STC:01:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
                          'STC:02:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
        },
    'PF2K4:WFS:':{
            'MMS:STATE:PMPS:ARB:ENABLE_RBV':{'Thresholds':None, 'Filter':None},
                          'STC:01:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
                          'STC:02:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
        },
    'TM1K4:ATM:':{
            'MMS:STATE:PMPS:ARB:ENABLE_RBV':{'Thresholds':None, 'Filter':None},
                          'STC:01:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
        },
    'TM2K4:ATM:':{
            'MMS:STATE:PMPS:ARB:ENABLE_RBV':{'Thresholds':None, 'Filter':None},
                          'STC:01:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
        },
    'ST1K4:TEST:':{
                        'MMS:STATE:ERR_RBV':{'Thresholds':None, 'Filter':None},
                        'MMS:STATE:GET_RBV':{'Thresholds':None, 'Filter':'PPS:NEH1:1:ST3K4INSUM!=0'},
                 'MMS:STATE:ST3K4_AUTO_RBV':{'Thresholds':None, 'Filter':None},
        },
    'SP1K4:':{
            'FZP:STATE:PMPS:ARB:ENABLE_RBV':{'Thresholds':None, 'Filter':None},
                   'PPM:CAM:TSS:SyncStatus':{'Thresholds':None, 'Filter':'SP1K4:PPM:CAM:TriggerMode_RBV!=0'}, 
                 'PPM:CAM:TSS:DiffVsIntAvg':{'Thresholds':{'LOLO':-1.5,'LOW':-1.5,'HIGH':1.5,'HIHI':1.5}, 'Filter':None},        
        },
    'TMO:SPEC:':{
                          'RTD:01:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
                          'RTD:02:TEMP_RBV':{'Thresholds':{'LOLO':0,'LOW':5,'HIGH':70,'HIHI':85}, 'Filter':None},
                              'FWM:VAL_RBV':{'Thresholds':{'LOLO':3.5,'LOW':3.7,'HIGH':5.0,'HIHI':6.0}, 'Filter':None},
                          
        },
    'TMO:IP1:':{
                     'MAIN:GAUGE:PRESS_RBV':{'Thresholds':{'LOLO':0,'LOW':0,'HIGH':1E-7,'HIHI':1E-6}, 'Filter':None},
        },
    }

class Alarm():
    def __init__(self, alarms):
        self.alarm_dict = alarms
        
    def check_pvs(self, display=False):
        status = True
        for device in self.alarm_dict.keys():
            for PV in self.alarm_dict[device].keys():
                sig = EpicsSignal( ('%s%s' % (device, PV) ) )
                time.sleep(0.5)
                if sig.connected:
                    status = status and True
                    if display:
                        val = sig.get()
                        logger.info( ('%s%s is connected and has value %s' % (device, PV, val) ) )
                else:
                    status = status and False
                    logger.warning( ('%s%s is not connected!' % (device, PV) ) )
        return status
    
    def set_thresholds(self):
        if self.check_pvs():
            logger.info('All PVs are connected.')
            for device in self.alarm_dict.keys():
                for PV in self.alarm_dict[device].keys():
                    if self.alarm_dict[device][PV]['Thresholds'] is not None:
                        for key in ['LLSV','HHSV']:
                            sig = EpicsSignal( ('%s%s.%s' % (device, PV, key) ) )
                            time.sleep(0.2)
                            sig.put(2)
                        for key in ['LSV','HSV']:
                            sig = EpicsSignal( ('%s%s.%s' % (device, PV, key) ) )
                            time.sleep(0.2)
                            sig.put(1)
                        for key in self.alarm_dict[device][PV]['Thresholds']:
                            sig = EpicsSignal( ('%s%s.%s' % (device, PV, key) ) )
                            time.sleep(0.5)
                            if sig.connected:
                                prev_val = sig.get()
                                new_val = self.alarm_dict[device][PV]['Thresholds'][key]
                                sig.put(new_val)
                                if prev_val != new_val:
                                    logger.info( ('changed %s%s.%s from %s to %s') % (device, PV, key, prev_val, new_val) )
                                
                            else:
                                logger.error(('%s%s.%s is not connected!' % (device, PV, key) ))
        else:
            logger.error('some PVs are not connected!')

    def show_filters(self):
        for device in self.alarm_dict.keys():
            for PV in self.alarm_dict[device].keys():
                if self.alarm_dict[device][PV]['Filter'] is not None:
                    print( ('%s%s filter: %s' % (device, PV, self.alarm_dict[device][PV]['Filter']) ) )

tmo_alarm_utility = Alarm(alarm_dict)
    
            
        