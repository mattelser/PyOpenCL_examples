# PyOpenCL examples
Just a few demos I found useful to create while learning PyOpenCL

## Contents

- `sineOfTheTimes1d.py` - operating on a flattened array, using only 
  `array_to_device` to handle the buffers and mem copy
- `sineOfTheTimes.py` - use `Image` memory object and more manually 
  manage memory. Plus this cool output:
  ![sine o' the times](./sot.gif)

## requirements
requires the following modules:
- `numpy`
- `pyopencl`
- `opencv-python` for display purposes
