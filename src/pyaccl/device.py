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
from pyaccl.scan import scan_overlay

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
    def __init__(self, xclbin, board_idx=0, cclo_idx=0):
        self.local_alveo = pynq.Device.devices[board_idx]
        self.ol = pynq.Overlay(xclbin, device=self.local_alveo)
        accl_dict = scan_overlay(self.ol)
        self.cclo = self.ol.__getattr__(accl_dict[cclo_idx])
        self.hostctrl = [self.ol.__getattr__(c) for c in accl_dict[cclo_idx]["controllers"]]
        self.mmio = self.cclo.mmio
        self.protocol = accl_dict[cclo_idx]["poe"]["protocol"]
        self.devicemem = self.ol.__getattr__(accl_dict[cclo_idx]["memory"][0])
        self.rxbufmem = [self.ol.__getattr__(b) for b in accl_dict[cclo_idx]["memory"]]
        if accl_dict[cclo_idx]["poe"] is None:
            self.networkmem = None
        elif self.protocol == "TCP":
            self.networkmem = []
            self.networkmem.append([self.ol.__getattr__(b) for b in accl_dict[cclo_idx]["poe"]["memory"][0]])
            self.networkmem.append([self.ol.__getattr__(b) for b in accl_dict[cclo_idx]["poe"]["memory"][1]])
        else:
            self.networkmem = None

        print("AlveoDevice connected")

    def read(self, offset):
        return self.mmio.read(offset)

    def write(self, offset, val):
        return self.mmio.write(offset, val)

    def call(self, scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2):
        if self.hostctrl is not None:
            self.hostctrl[0].call(scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2)
        else:
            raise Exception("Host calling not supported, no hostctrl found")

    def start(self, scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2):
        if self.hostctrl is not None:
            return self.hostctrl[0].start(scenario, count, comm, root_src_dst, function, tag, arithcfg, compression_flags, stream_flags, addr_0, addr_1, addr_2)
        else:
            raise Exception("Host calling not supported, no hostctrl found")
