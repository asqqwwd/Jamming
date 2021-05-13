import settings, time
from utils.pool import PoolBlockGet, PoolBlockPut, PoolNoBlock, PoolCycle

raw_input_pool = PoolBlockGet(int(settings.IN_FS * settings.NOISE_LENGTH))
processed_input_pool = PoolBlockGet(
    int(settings.IN_FS * settings.KWS_FRAME_LENGTH))
last_input_pool = PoolNoBlock()
noise_pool = PoolBlockPut(settings.OUT_FRAMES_PER_BUFFER)
keyword_pool = PoolNoBlock()
kws_cache_pool = PoolCycle(int(settings.OUT_FS * 4))
test_pool = PoolCycle()

start_time = time.time()  # 系统实际开始运行时间
run_time = 0  # 使用input记录出的时间