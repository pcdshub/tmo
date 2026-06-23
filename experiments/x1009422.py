import matplotlib.pyplot as plt
from tmo.db import camviewer
from pcdsdevices.beam_stats import BeamEnergyRequest
from ophyd.device import Component as Cpt
from ophyd.device import FormattedComponent as FCpt
from ophyd.signal import EpicsSignal, EpicsSignalRO
from pcdsdevices.device import Device
from pcdsdevices.interface import BaseInterface
from ophyd.pv_positioner import PVPositioner, PVPositionerIsClose
from pcdsdevices.interface import FltMvInterface


def get_fzp_screen_grab():
   
    cam_dict={
        "im5k4": {
            "roi": {"w":(175,363), "h":(84,355)},
            "markers": [(248,251), (293,252), (249,295), (291,296)]
                },
        "sp1k4-gige": {
            "roi": {"w":(142,425), "h":(280,350)},
            "markers": [(205,328), (205,296), (417,328), (417,296)]
                },
        "sp1k4_yag_f": {
            "roi": {"w":(285,372), "h":(212,250)},
            "markers": [(309,237), (336,237), (309,226), (336,226)]
                }
        }
    
    def get_im(cam_name):
        cam_obj = getattr(camviewer, cam_name)
        im = cam_obj.image2.array_data.get()
        return im.reshape(cam_obj.image2.array_size.height.get(), cam_obj.image2.array_size.width.get())
    
    fig, axs = plt.subplots(1,3)
    for i, cam in enumerate(['im5k4', 'sp1k4-gige', 'sp1k4_yag_f']):
        im = get_im(cam)
        axs[i].imshow(im)
        axs[i].set_xlim(cam_dict[cam]['roi']['w'])
        axs[i].set_ylim(cam_dict[cam]['roi']['h'])
        for j in range(4):
            axs[i].plot(cam_dict[cam]['markers'][j][0], cam_dict[cam]['markers'][j][1], marker='+', ms=20)

class MagnetMotor(FltMvInterface, PVPositionerIsClose):
        setpoint = FCpt(EpicsSignal, 'RIX:USER:MCC:EPHOTK:SET2', kind = 'hinted')
        readback = FCpt(EpicsSignalRO, 'SIOC:SYS0:ML01:AO809', kind = 'hinted')
        atol = 0.1
        rtol = 0


class User():
    und_delay = BeamEnergyRequest('RIX', name='und_delay' ,atol=0, skip_small_moves=False, pv_index=1)
    sxrss_delay = MagnetMotor('', name='sxrss_delay')
        
       