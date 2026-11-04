from tmo.db import RE, bp
#from ophyd.device import Component as Cpt
from pcdsdevices.device import ObjectComponent as OCpt
#from ophyd.device import FormattedComponent as FCpt
from bluesky.callbacks import LivePlot
from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface


class PlotClass(BaseInterface,Device):
    """ This class overwrites some of the native functions in the device to only look at new data. """  
    def describe(self):
        return self.device.describe()
    
    def stage(self):
        def read_data(*, old_value, value, timestamp, **kwargs):
            self.data={'value':value, 'timestamp':timestamp}
            self._new_data=1
        self.uid = self.device.subscribe(read_data)
        
    def trigger(self):
        self.data={}
        self._new_data=0
        return self.device.trigger()
        
    def read(self):
        while not self._new_data: continue
        return{self.device.name:self.data}

    def unstage(self):
        self.device.unsubscribe(self.uid)


def plot_it(device, num_samples=100):
    """
    Use Bluesky run engine to plot some readings from a device. 

    Parameters
    ----------
    device : object
        Device to read.
    num_samples : int, optional
        Number of samples to record. The default is 100.

    """
    
    PlotItClass = type('PlotItClass',(PlotClass, BaseInterface, Device), {'device': OCpt(device, kind='hinted')})
    dev=PlotItClass(name=device.name)
    RE(bp.count([dev],num=num_samples), LivePlot(device.name))
    
    