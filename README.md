# fenrir

## Basic Event Visualization Demo

SSH onto fenrir with X forwarding enabled

```
$ ssh -X ncs@fenrir
$ cd Eben/fenrir
```

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
ncs@fenrir:~/Eben/fenrir$ build/SlamDemo --calibration-json fenrir_calibration.json --hot-pixels-dir hot_pixels_fenrir --sbm-num-threads 12  --max-events-in-buffer 24000
```

`--sbm-num-threads` is one of many adjustable flags, in this case controlling the number of threads dispatched for each stereo block matching operation.

If the executable does not exist for some reason, rebuild it as follows.

## Depth Perception Demo Rendered on Client

Rendering on fenrir and sending the GUI over SSH creates unnecessary overhead. Run the client script from the directory of this repository

```
python scripts/fenrir_vis_client.py
```

The dependencies for the above script are `numpy`, `opencv-python`, and `zmq`. The script can write a video if desired:

```
python scripts/fenrir_vis_client.py --record_video --video_name <DEFAULT=recording.mp4>
```

The video will be saved after you interrupt the application with `Ctrl+C`. Run the demo on the robot but now pointing it towards the local IP address of the client; if running from hugin:

```
ncs@fenrir:~/Eben/fenrir$ build/SlamDemo --calibration-json fenrir_calibration.json --hot-pixels-dir hot_pixels_fenrir --sbm-num-threads 12  --max-events-in-buffer 24000 --client-addr 172.16.222.30
```

#### Starting the demo on the robot before opening the client listener will cause depth frames to backlog on the robot, which will look really weird when you run the visualization.

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

5) ZeroMQ (`sudo apt install libzmq3-dev cppzmq-dev`)