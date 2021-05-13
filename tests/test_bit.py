import numpy as np
def decimal2binary(num):
    return ["{:b}".format(x) for x in num]
def run():
    a = np.random.random(2)
    print(a)
    b = (a * 2**15).clip(-2**15, 2**15 - 1).astype(np.int16).tobytes()
    print(decimal2binary(b))
    pass

if __name__=="__main__":
    run()