# /*******************************************************************************
#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# *******************************************************************************/

import pynq
import numpy as np
import math

class ACCLBuffer(np.ndarray):
    next_free_address = 0
    def __new__(cls, shape, dtype=np.float32, target=None, zmqsocket=None, physical_address=None, prealloc=True):
        if zmqsocket is None:
            self = pynq.allocate(shape, dtype=dtype, target=target)
        else:
            self = super().__new__(cls, shape, dtype=dtype)

            if physical_address is None:
                self.physical_address = ACCLBuffer.next_free_address
                # allocate on 4K boundaries
                # not sure how realistic this is, but it does help
                # work around some addressing limitations in RTLsim
                ACCLBuffer.next_free_address += math.ceil(self.nbytes/4096)*4096

            # Allocate buffer to prevent emulator from writing out of bounds
            self.socket = zmqsocket
            self.prealloc = prealloc
            if self.prealloc:
                self.preallocate_memory()

        return self

    def __array_finalize__(self, obj):
        if isinstance(obj, ACCLBuffer):
            self.socket = obj.socket
            self.prealloc = obj.prealloc
            self.target = obj.target
            self.physical_address = obj.physical_address
        else:
            self.socket = None
            self.prealloc = False
            self.target = None
            self.physical_address = None

    # Devicemem read request  {"type": 2, "addr": <uint>, "len": <uint>}
    # Devicemem read response {"status": OK|ERR, "rdata": <array of uint>}
    def sync_from_device(self):
        """Copy buffer data in the device to host direction
        """        
        if self.socket is None:
            super().sync_from_device()
        else:
            self.socket.send_json({"type": 2, "addr": self.physical_address, "len": self.nbytes})
            ack = self.socket.recv_json()
            assert ack["status"] == 0, "ZMQ mem buffer read error"
            self.view(np.uint8)[:] = ack["rdata"]

    # Devicemem write request  {"type": 3, "addr": <uint>, "wdata": <array of uint>}
    # Devicemem write response {"status": OK|ERR}
    def sync_to_device(self):
        """Copy buffer data in the host to device direction
        """  
        if self.socket is None:
            super().sync_to_device()
        else:
            self.socket.send_json({"type": 3, "addr": self.physical_address, "wdata": self.view(np.uint8).tolist()})
            ack = self.socket.recv_json()
            assert ack["status"] == 0, "ZMQ mem buffer write error"

    # Devicemem allocate request  {"type": 4, "addr": <uint>, "len": <uint>}
    # Devicemem allocate response {"status": OK|ERR}
    def preallocate_memory(self):
        self.socket.send_json({"type": 4, "addr": self.physical_address, "len": self.nbytes})
        ack = self.socket.recv_json()
        assert ack["status"] == 0, "ZMQ mem buffer allocation error"

    @property
    def device_address(self):
        """Get physical address in FPGA memory

        Returns:
            int: Physical address
        """        
        if self.socket is None:
            return self.device_address
        else:
            return self.physical_address

    def __del__(self):
        if self.socket is None:
            self.freebuffer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.socket is None:
            self.free_buffer()  
        return 0
