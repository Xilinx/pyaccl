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

import zmq
import pynq

class SimMMIO():
    def __init__(self, zmqsocket):
        self.base_addr = 0
        self.socket = zmqsocket

    # MMIO read request  {"type": 0, "addr": <uint>}
    # MMIO read response {"status": OK|ERR, "rdata": <uint>}
    def read(self, offset):
        self.socket.send_json({"type": 0, "addr": offset})
        ack = self.socket.recv_json()
        assert ack["status"] == 0, "ZMQ MMIO read error"
        return ack["rdata"]

    # MMIO write request  {"type": 1, "addr": <uint>, "wdata": <uint>}
    # MMIO write response {"status": OK|ERR}
    def write(self, offset, val):
        self.socket.send_json({"type": 1, "addr": offset, "wdata": val})
        ack = self.socket.recv_json()
        assert ack["status"] == 0, "ZMQ MMIO write error"

class SimDevice():
    def __init__(self, zmqadr="tcp://localhost:5555"):
        print("SimDevice connecting to ZMQ on", zmqadr)
        self.socket = zmq.Context().socket(zmq.REQ)
        self.socket.connect(zmqadr)
        self.mmio = SimMMIO(self.socket)
        self.devicemem = None
        self.rxbufmem = None
        self.networkmem = None
        print("SimDevice connected")

    # Call request  {"type": 5, arg names and values}
    # Call response {"status": OK|ERR}
    def call(self, scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2):
        self.socket.send_json({ "type": 5,
                                "scenario": scenario,
                                "count": count,
                                "comm": comm,
                                "root_src_dst": root_src_dst,
                                "function": function,
                                "tag": tag,
                                "arithcfg": arithcfg,
                                "compression_flags": compression_flags,
                                "stream_flags": stream_flags,
                                "addr_0": addr_0,
                                "addr_1": addr_1,
                                "addr_2": addr_2})
        ack = self.socket.recv_json()
        assert ack["status"] == 0, "ZMQ call error"

    def start(self, scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2):
        self.socket.send_json({ "type": 5,
                                "scenario": scenario,
                                "count": count,
                                "comm": comm,
                                "root_src_dst": root_src_dst,
                                "function": function,
                                "tag": tag,
                                "arithcfg": arithcfg,
                                "compression_flags": compression_flags,
                                "stream_flags": stream_flags,
                                "addr_0": addr_0,
                                "addr_1": addr_1,
                                "addr_2": addr_2})
        return self

    def read(self, offset):
        return self.mmio.read(offset)

    def write(self, offset, val):
        return self.mmio.write(offset, val)

    def wait(self):
        ack = self.socket.recv_json()
        assert ack["status"] == 0, "ZMQ call error"

class AlveoDevice():
    def __init__(self, overlay, cclo_ip, hostctrl_ip, mem=None, board_idx=0):
        self.ol = overlay
        self.cclo = cclo_ip
        self.hostctrl = hostctrl_ip
        self.mmio = self.cclo.mmio
        self.local_alveo = pynq.Device.devices[board_idx]
        if mem is None:
            print("Best-effort attempt at identifying memories to use for RX buffers")
            if self.local_alveo.name == 'xilinx_u250_gen3x16_xdma_shell_3_1':
                print("Detected U250 (xilinx_u250_gen3x16_xdma_shell_3_1)")
                self.devicemem   = self.ol.bank1
                self.rxbufmem    = [self.ol.bank0, self.ol.bank1, self.ol.bank2]
                self.networkmem  = self.ol.bank3
            elif self.local_alveo.name == 'xilinx_u250_xdma_201830_2':
                print("Detected U250 (xilinx_u250_xdma_201830_2)")
                self.devicemem   = self.ol.bank0
                self.rxbufmem    = self.ol.bank0
                self.networkmem  = self.ol.bank0
            elif self.local_alveo.name == 'xilinx_u280_xdma_201920_3':
                print("Detected U280 (xilinx_u280_xdma_201920_3)")
                self.devicemem   = self.ol.HBM0
                self.rxbufmem    = [self.ol.HBM0, self.ol.HBM1, self.ol.HBM2, self.ol.HBM3, self.ol.HBM4, self.ol.HBM5]
                self.networkmem  = self.ol.HBM6
        else:
            print("Applying user-provided memory config")
            self.devicemem = mem[0]
            self.rxbufmem = mem[1]
            self.networkmem = mem[2]
        print("AlveoDevice connected")

    def read(self, offset):
        return self.mmio.read(offset)

    def write(self, offset, val):
        return self.mmio.write(offset, val)

    def call(self, scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2):
        if self.hostctrl is not None:
            self.hostctrl.call(scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2)
        else:
            raise Exception("Host calling not supported, no hostctrl found")

    def start(self, scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2):
        if self.hostctrl is not None:
            return self.hostctrl.start(scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2)
        else:
            raise Exception("Host calling not supported, no hostctrl found")
