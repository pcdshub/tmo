from epics import caget, caput, cainfo
from time import sleep
def _setVoltage(V_set):
    if V_set < 2:
        caput("TMO:IP1:HMP:01:2:SELECT", 0)
        caput("TMO:IP1:HMP:01:3:SELECT", 0)
        caput("TMO:IP1:HMP:01:4:SELECT", 0)
    else:
        caput("TMO:IP1:HMP:01:2:SET_VOLT", V_set)
        caput("TMO:IP1:HMP:01:2:SELECT", 1)
        caput("TMO:IP1:HMP:01:3:SET_VOLT", V_set)
        caput("TMO:IP1:HMP:01:3:SELECT", 1)
        caput("TMO:IP1:HMP:01:4:SET_VOLT", V_set)
        caput("TMO:IP1:HMP:01:4:SELECT", 1)

def setMBEScurrent(current):
    max_curr = 3.0
    coil_resist = 32
    caput("TMO:IP1:HMP:01:2:SET_CURR", max_curr)
    caput("TMO:IP1:HMP:01:3:SET_CURR", max_curr)
    caput("TMO:IP1:HMP:01:4:SET_CURR", max_curr)
    caput("TMO:IP1:HMP:01:1:SET_VOLT", 64.0) #set max voltage on first channel
    
    V_tot = current*coil_resist
    V_set = (V_tot-10)/3
    print(V_set)
    _setVoltage(V_set)
    
    caput("TMO:IP1:HMP:01:1:SET_CURR", current)
    sleep(0.2)
    V1 = caget("TMO:IP1:HMP:01:1:VOLT")
    retI = caget("TMO:IP1:HMP:01:1:CURR")
    
    V_t = V1+V_set*3
    coil_resist_act = (V_t)/retI
    V_t = coil_resist_act*current
    V_set = (V_t-10)/3
    print(V_set)
    _setVoltage(V_set)
    
