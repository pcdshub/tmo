from epics import PV
from time import sleep
import numpy as np

def collect(pv='IM1K0:XTES:CAM:IMAGE2:ArrayData', num_img=10):
    print('collecting')
    obj = PV(pv)
    time = 0.10
    images = []
    while len(images) < num_img:
        img = obj.get()
        if len(images) == 0:
            images.append(img)
        if not np.array_equal(images[-1], img):
            print('getting an image ', len(images))
            images.append(img)
            sleep(time)
        else:
            sleep(0.01)

    print('collected images ', images)

if __name__ == '__main__':
    collect()
