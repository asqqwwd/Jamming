import threading, os
import numpy as np
# from pocketsphinx import DefaultConfig, Decoder, get_model_path, get_data_path
from scipy import signal

import global_var
from utils.resampler import Resampler
from utils.codec import Codec


class KeywordSpotting(threading.Thread):
    def __init__(self, in_fs, out_fs, mute_period_length, kws_frame_length):
        threading.Thread.__init__(self)
        # 初始化配置
        self.daemon = True
        self.exit_flag = False
        self.in_fs = in_fs
        self.out_fs = out_fs
        self.mute_period_frames_count = int(in_fs * mute_period_length)
        self.kws_frames_count = int(in_fs * kws_frame_length)
        model_path = get_model_path()
        config = Decoder.default_config()
        config.set_string('-hmm', os.path.join(model_path, 'en-us'))  # 声学模型路径
        config.set_string('-dict',
                          os.path.join(model_path,
                                       'cmudict-en-us.dict'))  # 字典路径
        config.set_string('-keyphrase', 'alexa')
        config.set_float('-kws_threshold', 1e-20)
        config.set_string('-logfn', './logs/tmp')  # INFO输出到其他位置
        self.decoder = Decoder(config)
        self.decoder.start_utt()

        self.start()

    def run(self):
        while not self.exit_flag:
            # 1.从input池中读取一定长度的数据。该过程可能被阻塞，直到池中存在足够多数据。
            processed_input_frames = global_var.processed_input_pool.get(
                self.kws_frames_count)
            # 2.如果检测出该数据段中存在关键字，则对该数据进行重采样，填充后，存入keyword池
            speak_start_end = self._kws(processed_input_frames)
            if speak_start_end:
                global_var.kws_cache_pool.put(processed_input_frames)
                out_data = global_var.kws_cache_pool.get(
                    (speak_start_end[1] - speak_start_end[0]) * 160 *
                    5)  # 这里的160是根据16000采样率和0.01秒计算出来的
                global_var.keyword_pool.put(
                    self._padding(
                        Resampler.resample(processed_input_frames, self.in_fs,
                                           self.out_fs), 0,
                        self.mute_period_frames_count))
            # 3.如果未检测出关键字，则将此段输入暂存
            else:
                # 缓存输入声音，数据输入池子中时，超过容量则丢弃前面的一部分数据
                global_var.kws_cache_pool.put(processed_input_frames)

    def stop(self):
        self.exit_flag = True
        self.join()

    #改了返回值
    def _kws(self, frames):
        buf = Codec.encode_audio_to_bytes(frames, 1, 16)
        if buf:
            self.decoder.process_raw(buf, False, False)
            if self.decoder.hyp() != None:
                for seg in self.decoder.seg():
                    out_res = [seg.start_frame, seg.end_frame]
                    print([seg.word, seg.prob, seg.start_frame, seg.end_frame])
                self.decoder.end_utt()
                self.decoder.start_utt()
                return out_res
        return None

    def _padding(self, frames, padding_value, padding_num):
        res = np.pad(frames, (0, padding_num),
                     'constant',
                     constant_values=(padding_value, padding_value))
        return res