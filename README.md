# fenrir

## Basic Event Visualization Demo

There should exist a python virtual environment, activate with the following command:

```
ncs@fenrir:~/Eben/fenrir$ source ~/ebenvenv/bin/activate
```

Then to see events:

```
python scripts/dv_processing_example.py --calib_json fenrir_calibration.json --hot_pixel_dir hot_pixels_fenrir
```

Pass the flag `--no_filters` to see unfiltered events.

## Depth Perception Demo

From the project directory, simply run the executable:

```
ncs@fenrir:~/Eben/fenrir$ build/SlamDemo --calibration-json fenrir_calibration.json --hot-pixels-dir hot_pixels_fenrir --sbm-num-threads 13
```

`--sbm-num-threads` is one of many adjustable flags, in this case controlling the number of threads dispatched for each stereo block matching operation.

If the executable does not exist for some reason, rebuild it as follows.

### Build Instructions

Assuming `gcc` and `cmake` are installed and working and that dependencies from the next secion have been installed (which is likely unless the system was wiped),

Make sure the build directory exists and enter it

```
ncs@fenrir:~/Eben/fenrir$ mkdir build
ncs@fenrir:~/Eben/fenrir$ cd build
```

Configure cmake to build an optimized executable and run the compilation:

```
ncs@fenrir:~/Eben/fenrir/build$ cmake -DCMAKE_BUILD_TYPE=Release ../
ncs@fenrir:~/Eben/fenrir/build$ make
```

Now the executable `SlamDemo` should be present in the `build` directory.

### Dependencies

The following should already be installed on the system, but if something breaks, re-installation may be necessary.

1) [cnpy](https://github.com/rogersce/cnpy)

2) OpenCV

3) [dv-processing](https://dv-processing.inivation.com/master/installation.html)

4) Boost (`sudo apt install libboost-all-dev`)