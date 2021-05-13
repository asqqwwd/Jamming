import pyaudio
import wave
import time
import sys
from utils.codec import Codec
from utils.modulate import Modulate
from utils.access import Access

tmp = bytes()

def run():
    filename = "./waves/modulated/offer_40k.wav"

    wf = wave.open(filename, 'rb')
    print(wf.getparams())

    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()

    st = time.time()

    def callback(in_data, frame_count, time_info, status):
        global tmp
        tmp += in_data
        if time.time()-st>10:
            Access.save_wave(tmp, "./tmp.wav", 1, 2, 96000)
            raise ValueError("End")
        print(max(in_data))
        return (None, pyaudio.paContinue)

    stream = p.open(format=p.get_format_from_width(2),
                    channels=1,
                    rate=96000,
                    input=True,
                    input_device_index=2,
                    frames_per_buffer=96000,
                    stream_callback=callback)

    # def callback(in_data, frame_count, time_info, status):
    #     tmp += in_data
    #     return (None, pyaudio.paContinue)

    # stream = p.open(format=p.get_format_from_width(2),
    #                 channels=wf.getnchannels(),
    #                 rate=wf.getframerate(),
    #                 output=True,
    #                 output_device_index=None,
    #                 stream_callback=callback)

    # start the stream (4)
    stream.start_stream()

    # wait for stream to finish (5)
    while stream.is_active():
        time.sleep(0.1)

    # stop stream (6)
    stream.stop_stream()
    stream.close()
    wf.close()

    # close PyAudio (7)
    p.terminate()