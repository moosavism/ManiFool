import pycuda.driver as drv

class Holder(drv.PointerHolderBase):
    """
    A simple class for connecting Pytorch tensors in GPU to pycuda arrays, which
    can then be used for Pycuda functions in GPU.
    """

    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()
    def get_pointer():
        return self.t.data_ptr()
