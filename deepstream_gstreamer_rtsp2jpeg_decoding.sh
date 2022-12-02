gst-launch-1.0 -vvv rtspsrc location=rtsp://192.168.1.140:554/jpeg ! rtpjpegdepay ! jpegparse ! nvjpegdec ! nvdsosd ! nveglglessink sync=0
