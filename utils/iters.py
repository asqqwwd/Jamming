class SlideWindow():
    def __init__(self, frames, window_length, hop_length):
        self.frames = frames
        self.window_length = int(window_length)
        self.hop_length = int(hop_length)

    # 迭代对象最简单写法，无需迭代器。index自动从0开始递增
    def __getitem__(self, index):
        sti = index * self.hop_length
        edi = sti + self.window_length
        if edi > len(self.frames):
            raise IndexError()
        return self.frames[sti:edi]
