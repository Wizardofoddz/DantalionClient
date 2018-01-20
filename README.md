# DantalionClient
The Dantalion client is a python application used to communicate with one or more Dantalion endpoints via MQTT.  It would be pretty easy to hack it up to accept image streams from other sources, and I'm wide open for pull requests to integrate other data sources.   This implementation provides the ability to view raw camera streams, to calculate the camera matrix and lens distortion instrinsics, and the ability to see an interframe differential image stream.  The architecture is based on the IDEF model I use to allow visual processing pipelines to be easily assembled from pieces.
![Raw Mode](../master/doco/Raw%20User%20View.png?raw=true "Raw Mode Images")
The current calibration is based on a 9x7 checkerboard, where 24 samples are collected with a minimum inter-sample distance of 22 pixels assuming a 640x480 image.  Once enough samples are collected, the system will transmit the computed camera intrinsics to Dantalion via MQTT and a new rectified image channel will be available from Dantalion.  With an alpha of 1.0, these images look like this.

![Raw Mode](../master/doco/Undistorted%20User%20View.png?raw=true "Undistorted Mode Images")
There is also a differential mode, where the images are defined as the absolute delta between frames.Here is an example image from that stream.
![Raw Mode](../master/doco/DifferentialScreen.png?raw=true "Interframe image delta")

Yes, this documentation needs more work.
