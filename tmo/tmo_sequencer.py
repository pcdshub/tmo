from pcdsdevices.sequencer import EventSequence, EventSequencer
from pcdsdevices.evr import Trigger
import logging

logger = logging.getLogger(__name__)

tmo_sequencer1 =  EventSequencer('ECS:SYS0:1', name='TMO')
tmo_sequencer8 = EventSequencer('ECS:SYS0:8', name='laser')
tmo_sequencer9 = EventSequencer('ECS:SYS0:9', name='spare')

tmo_evr0 = Trigger("TMO:MISC:EVR:01:TRIG0", name='gas jet')
tmo_evr1 = Trigger("TMO:MISC:EVR:01:TRIG1", name='scope trigger')
tmo_evr2 = Trigger("TMO:MISC:EVR:01:TRIG2", name='laser shutter')
tmo_evr3 = Trigger("TMO:MISC:EVR:01:TRIG3", name='TMO3')
tmo_evr4 = Trigger("TMO:MISC:EVR:01:TRIG4", name='TMO4')

las_evr1 = Trigger("EVR:LAS:LHN:04:TRIG1", name='fs_laser_on_time')
las_evr2 = Trigger("EVR:LAS:LHN:04:TRIG2", name='fs_laser_goose')

#Common functions to use with Sequencer. 
def gas_jet_sequence(num_on=12, num_off=1):
    """
    Setup sequencer 1 to modulate the pulsing of the gas jet. This starts the sequencer with a sequence of event code 70's and sets trigger 0 of TMO EVR to trigger on EC 70. 
    Parameters:
    -------------
    num_on: number of shots with the gas jet
    num_off: number of shots with no gas jet
    """
    if ( isprime(num_on + num_off) is False ):
        print ("WARNING: The number of steps in this sequence is not a prime number. This could lead to systematics in your data.") 
    seq = [];
    step = 0
    while (step < num_on-1) :
        seq.append([70,1,0,0])
        step = step + 1
    seq.append([70,1+num_off,0,0])

    tmo_sequencer1.sequence.put_seq(seq)
    tmo_sequencer1.start()
    tmo_evr0.eventcode.set(70)
    tmo_evr0.enable()


def goose_trigger(duck=4, goose=3):
    """
    Setup sequencer 8 to run the goose trigger. 
    Parameters:
    --------------
    duck: number of shots where the laser is timed with the x-rays
    goose: number of shots where the laser is mis-time with the x-rays
    """
    if ( isprime(duck + goose) is False ):
        print ("WARNING: The number of steps in this sequence is not a prime number. This could lead to systematics in your data.") 
    seq = [];
    step = 0
    while (step < duck):
        seq.append([67,1,0,0])
        step = step + 1
    step = 0
    while (step < goose): 
         seq.append([68,1,0,0])
         step = step + 1
    
    tmo_sequencer8.sequence.put_seq(seq)
    tmo_sequencer8.start()


def goose_trigger_configure(delay=None):
    """
    Configure the laser evr (EVR:LAS:LHN:04) for the TMO goose trigger
    Parameters:
    ---------------
    delay: optional parameter to adjust the delay of the goosed trigger. The delay is in ns, and is relative to the timed shots.
           positive values indicate that the goose trigger is delayed wrt the timed shots. Not giving a value will leave at the current setting. 
    """
    las_evr1.eventcode.set(67)
    las_evr2.eventcode.set(68)

    if (delay != None):
        t0 = las_evr1.ns_delay.get()
        las_evr2.ns_delay.set( t0+delay )
    goose_trigger(duck=1, goose=0)


def burst_trigger(burst_on=1, burst_off=0):
    if ( isprime(burst_on + burst_off) is False ):
        print ("WARNING: The number of steps in this sequence is not a prime number. This could lead to systematics in your data.") 
    seq = [];
    step = 0
    while (step < burst_on - 1):
        seq.append([71,1,0,1])
        step = step + 1
    seq.append([71,1+burst_off,0,1])
    tmo_sequencer1.sequence.put_seq(seq)
    #tmo_sequencer1.start()

def get_tmo_sequencer_status():
    plst1 = tmo_sequencer1.play_status.value
    plst8 = tmo_sequencer8.play_status.value
    plst9 = tmo_sequencer9.play_status.value

    if (plst1 > 0):
        print('tmo_sequencer1 is running....')
        tmo_sequencer1.sequence.show()
    else:
        print('tmo_sequencer1 is not running....')

    if (plst8 > 0):
        print('tmo_sequencer8 is running....')
        tmo_sequencer8.sequence.show()

    if (plst9 > 0):
        print('tmo_sequencer9 is running....')
        tmo_sequencer9.sequence.show()

    print('No other sequencers are running in TMO.')

def isprime(num):
    for n in range(2,int(num**0.5)+1):
        if num%n==0:
            return False
    return True

