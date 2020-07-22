import pyopencl as cl
import pyopencl.array as pycl_array
import numpy as np
import time
import cv2


class SineOfTheTimes(object):
    def __init__(self):
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)

        self.WIDTH = 1280
        self.HEIGHT = 720

        self.program = cl.Program(self.context, f"""
        __kernel void getSineOfTheTimes(write_only image2d_t dest, 
                                       __global float *t)
        {{
          int x = get_global_id(0);
          int y = get_global_id(1);
          int2 pos = (int2)(x, y);
          
          // -- do some trig for fun visuals
          // bound t to [-pi, pi]
          float tBounded = fmod((double)*t, 6.2831853) - 3.1415926;
          int xB = {self.WIDTH/2} - x;
          int yB = {self.HEIGHT/3} - y;
          int xG = {self.WIDTH/3} - x;
          int yG = {2*self.HEIGHT/3} - y;
          int xR = {2*self.WIDTH/3} - x;
          int yR = {2*self.HEIGHT/3} - y;
          // get sine and scale it
          float sot = fabs(sin(tBounded * .002));
          // offset it for each color channel, make
          // it a circle, and bound it to the color 
          // channel max
          uint sotB = fmod(sot * (xB*xB + yB*yB), 255);
          uint sotG = fmod(sot * (xG*xG + yG*yG), 255);
          uint sotR = fmod(sot * (xR*xR + yR*yR), 255);

          // -- set the color of the pixel
          // opencv uses BGR for some reason
          uint4 col = {{sotB, sotG, sotR, 255}};
          write_imageui(dest, pos, col);
        }}
        """).build()

        self.startTime = time.time()
        self.frames = 0

    def getFrame(self):
        # grab the least sig 1000 digits from the current time
        # create as numpy singleton so it's typed correctly
        t = pycl_array.to_device(self.queue,
                                 np.float32(time.time() % 1000))

        # create a cpu-side array we'll fill
        out = np.empty((self.HEIGHT, self.WIDTH, 4), np.uint8)

        # create a partner for that array in gpu mem
        fmt = cl.ImageFormat(cl.channel_order.RGBA,
                             cl.channel_type.UNSIGNED_INT8)
        out_gpu = cl.Image(self.context,
                           cl.mem_flags.WRITE_ONLY,
                           fmt,
                           shape=(self.WIDTH,self.HEIGHT))

        # run the computation on the gpu
        self.program.getSineOfTheTimes(self.queue,
                                  (self.WIDTH, self.HEIGHT),
                                  None,
                                  out_gpu,
                                  t.data)
        # copy the results
        cl.enqueue_copy(self.queue,
                        out,
                        out_gpu,
                        origin=(0,0),
                        region=(self.WIDTH, self.HEIGHT))
        # self.queue.finish() if is_blocking=False in enqueue_copy
        return out

    def run(self):
        fps = 0
        window = cv2.namedWindow("sine of the times")
        arr = np.zeros((self.HEIGHT, self.WIDTH, 4), np.uint8)
        cv2.imshow(window, arr)
        cv2.waitKey(1)
        while True:
            fpsClockStart = time.time()
            frames = 0
            for _ in range(30):
                arr = self.getFrame()
                frames += 1

                # make text easier to read by using
                # white text with black border, i.e.

                cv2.putText(arr,
                            f'fps: {fps:.2f}',
                            (10,self.HEIGHT-10),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (0, 0, 0),
                            4)
                cv2.putText(arr,
                            f'fps: {fps:.2f}',
                            (10,self.HEIGHT-10),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (255, 255, 255),
                            1)

                # check if the window has been closed
                # and exit if so
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return
                cv2.imshow(window, arr)
                k = cv2.waitKey(1)
                if k == 27:
                    # escape key has been pressed
                    return
            fps = frames / (time.time() - fpsClockStart)


if __name__ == "__main__":
    sot = SineOfTheTimes()
    sot.run()
