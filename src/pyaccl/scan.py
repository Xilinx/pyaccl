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

from pynq import Overlay
import sys
import json

CCLO_VLNV = "Xilinx:ACCL:ccl_offload:1.0"
HOSTCONTROLLER_VLNV = "xilinx.com:ACCL:hostctrl:1.0"
ARBITER_VLNV = "xilinx.com:ACCL:client_arbiter:1.0"
UDP_VLNV = "xilinx.com:kernel:networklayer:1.0"
UDP_CMAC0_VLNV = "xilinx.com:kernel:cmac_0:1.0"
UDP_CMAC1_VLNV = "xilinx.com:kernel:cmac_1:1.0"
TCP_VLNV = "ethz.ch:kernel:network_krnl:1.0"
TCP_CMAC_VLNV = "ethz.ch:kernel:cmac_krnl:1.0"

def get_command_stream_ids(ip):
    ret = []
    # get the IPs streams (if none, return empty)
    try:
        stream_dict = ip["streams"]
    except:
        return ret
    # two types of IPs have command streams
    if ip["type"] == ARBITER_VLNV:
        for key in stream_dict.keys():
            if(key == "cmd_clients_0" or key == "cmd_clients_1"):
                ret.append(stream_dict[key]["stream_id"])
    if ip['type'] == CCLO_VLNV:
        for key in stream_dict.keys():
            if(key == "s_axis_call_req"):
                ret.append(stream_dict[key]["stream_id"]) 

    return ret  

def get_network_stream_ids(ip):
    ret = []
    # get the IPs streams (if none, return empty)
    try:
        stream_dict = ip["streams"]
    except:
        return ret
    # CCLO IPs have network streams
    if ip['type'] == CCLO_VLNV:
        for key in stream_dict.keys():
            # return the ID of the TX data stream
            if(key == "m_axis_eth_tx_data"):
                ret.append(stream_dict[key]["stream_id"]) 

    return ret

def get_cmac_stream_ids(ip):
    ret = []
    # get the IPs streams (if none, return empty)
    try:
        stream_dict = ip["streams"]
    except:
        return ret
    # network stack IPs have CMAC streams
    if ip['type'] == UDP_VLNV or ip['type'] == TCP_VLNV:
        for key in stream_dict.keys():
            # return the ID of the TX data stream
            if(key == "M_AXIS_nl2eth" or key == "axis_net_tx"):
                ret.append(stream_dict[key]["stream_id"])

    return ret  

def seek_controllers(ip_dict, stream_id):
    ret = []
    for key in ip_dict.keys():
        ip = ip_dict[key]
        if ip["type"] == ARBITER_VLNV:
            stream_dict = ip["streams"]
            for key in stream_dict.keys():
                if(stream_dict[key]["stream_id"] == stream_id):
                    # we've found a client arbiter, search upstream from it
                    for id in get_command_stream_ids(ip):
                        ret += seek_controllers(ip_dict, id)
        if ip['type'] == HOSTCONTROLLER_VLNV:
            stream_dict = ip["streams"]
            for key in stream_dict.keys():
                if(stream_dict[key]["stream_id"] == stream_id):
                    ret.append(ip["fullpath"])
    return ret

def seek_networkstack(ip_dict, stream_id):
    for key in ip_dict.keys():
        ip = ip_dict[key]
        if ip["type"] == UDP_VLNV or ip['type'] == TCP_VLNV:
            stream_dict = ip["streams"]
            for key in stream_dict.keys():
                if(stream_dict[key]["stream_id"] == stream_id):
                    # we've found a UDP/TCP stack
                    ret = {}
                    ret["name"] = ip["fullpath"]
                    ret["protocol"] = "TCP" if ip['type'] == TCP_VLNV else "UDP"
                    ret["memory"] = [ip["registers"]["axi00_ptr0"]["memory"], ip["registers"]["axi00_ptr1"]["memory"]] if ip['type'] == TCP_VLNV else []
                    return ret

def seek_cmac(ip_dict, stream_id):
    for key in ip_dict.keys():
        ip = ip_dict[key]
        if ip["type"] == UDP_CMAC0_VLNV or ip['type'] == UDP_CMAC1_VLNV or ip['type'] == TCP_CMAC_VLNV:
            stream_dict = ip["streams"]
            for key in stream_dict.keys():
                if(stream_dict[key]["stream_id"] == stream_id):
                    # we've found a CMAC stack
                    return ip["fullpath"]

def scan_overlay(ol):
    d = ol.ip_dict
    # search IP dictionary for CCLOs by VLNV
    cclo_dict = {}
    for key in d.keys():
        ip = d[key]
        if ip['type'] == CCLO_VLNV:
            # extract memory connections
            assert ip["registers"]["m_axi_0"] == ip["registers"]["m_axi_1"], "CCLO AXIMM ports not connected to same memory"
            cclo_dict[ip['fullpath']] = {}
            curr_cclo = cclo_dict[ip['fullpath']]
            curr_cclo["memory"] = ip["registers"]["m_axi_0"]
            # look for associated controllers
            curr_cclo["controllers"] = seek_controllers(d, get_command_stream_ids(ip))
            # look for associated network stacks and cmacs
            curr_cclo["networkstack"] = seek_networkstack(d, get_network_stream_ids(ip))
            curr_cclo["cmac"] = seek_cmac(d, d[curr_cclo["networkstack"]["name"]])

    return cclo_dict

def scan():
    # initialize overlay from xclbin file and scan
    ol = Overlay(sys.argv[1], download=False)
    s = scan_overlay(ol)
    with open(sys.argv[2],"w") as f:
        json.dump(s, f)