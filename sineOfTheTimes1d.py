import pyopencl as cl
import pyopencl.array as pycl_array
import numpy as np
import time
import cv2

class SineOfTheTimes1d(object):
    def __init__(self):
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)

        self.WIDTH = 720
        self.HEIGHT = 720

        self.program = cl.Program(self.context, f"""
        __kernel void sinOfTheTime(__global float *dest, __global const float *t)
        {{
          // it's 1d so we only get the x component
          int x = get_global_id(0);
          dest[x] = (sin(*t) * .5) + .5;
        }}
        """).build()

        self.startTime = time.time()
        self.frames = 0

    def pySinOfTheTime(self):
        t = pycl_array.to_device(self.queue, np.float32((time.time()) %6.2831))
        # create the output array as 1d vector and reshape later
        out = pycl_array.to_device(self.queue,
                                   np.ones((self.WIDTH*self.HEIGHT*3,),
                                   np.float32))
        self.program.sinOfTheTime(self.queue, out.shape, None, out.data, t.data)
        self.frames += 1
        return out.get().reshape((self.WIDTH, self.HEIGHT, 3))


if __name__ == "__main__":
    sot = SineOfTheTimes1d()
    try:
        while True:
            for _ in range(30):
                arr = sot.pySinOfTheTime()
                cv2.imshow("foo", arr)
                cv2.waitKey(1)
            print(f"fps: {sot.frames / (time.time() - sot.startTime)}")
    finally:
        print(f"fps: {sot.frames/(time.time() - sot.startTime)}")
