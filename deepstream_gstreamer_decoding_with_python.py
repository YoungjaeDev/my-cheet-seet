import subprocess as sp
import cv2
import numpy as np

cmd = ["gst-launch-1.0",
       "rtspsrc", "location=rtsp://admin:password@192.168.2.71/Streaming/Channels/1", "latency=100", "!",
       "queue", "!",
       "rtph264depay", "!",
       "h264parse", "!",
       "nvv4l2decoder", "drop-frame-interval=2", "!",
       "nvvideoconvert", "!",
       "video/x-raw,width=1920,height=1080,format=RGBA", "!",
       "queue", "!",
       "filesink", "location=/dev/stdout"
       ]

cmd2 = ["gst-launch-1.0",
       "rtspsrc", "location=rtsp://admin:password@192.168.2.72/Streaming/Channels/1", "latency=100", "!",
       "queue", "!",
       "rtph264depay", "!",
       "h264parse", "!",
       "nvv4l2decoder", "drop-frame-interval=2", "!",
       "nvvideoconvert", "!",
       "video/x-raw,width=1920,height=1080,format=RGBA", "!",
       "queue", "!",
       "filesink", "location=/dev/stdout"
       ]

gst1 = sp.Popen(cmd, stdout = sp.PIPE, bufsize=10, )
gst2 = sp.Popen(cmd2, stdout = sp.PIPE, bufsize=10, )

w=1920
h=1080
k = w*h
head_length = 528
# Including it will cause frame drift, so we need to remove the H.264 stream header first.
# Note header is added by camera producers and the header lenght could be different in different cameras. 
# Since I couldn't get the particular value by parsing something. The value here is obtained 
# by repeat trying and visualization. If someone know how to get it, please share it with us. Thanks so much!
gst1.stdout.read(head_length)
gst2.stdout.read(head_length)
while True:
    x = gst1.stdout.read(int(w*h*4))
    x = np.fromstring(x, dtype=np.uint8).reshape((h, w, 4))
    y = gst2.stdout.read(int(w*h*4))
    y = np.fromstring(y, dtype=np.uint8).reshape((h, w, 4))
    x = x[:, :, 0:3]
    y = y[:, :, 0:3]
    cv2.imshow("im1", x)
    cv2.imshow("im2", y)
    if cv2.waitKey(20) & 0xFF == ord('q'):
       break
@youngjae-avikus
