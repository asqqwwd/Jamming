import pyaudio

p = pyaudio.PyAudio()
for i in range(0,p.get_device_count()):
    dev_info = p.get_device_info_by_index(i)
    if "USB" in dev_info["name"]:
        print(dev_info)
