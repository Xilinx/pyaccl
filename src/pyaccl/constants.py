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

from enum import IntEnum, unique

@unique
class CCLOp(IntEnum):
    '''CCLO scenario IDs'''
    config         = 0
    copy           = 1
    combine        = 2
    send           = 3
    recv           = 4
    bcast          = 5
    scatter        = 6
    gather         = 7
    reduce         = 8
    allgather      = 9
    allreduce      = 10
    reduce_scatter = 11
    barrier        = 12
    all_to_all     = 13
    nop            = 255

@unique
class CCLOCfgFunc(IntEnum):
    '''CCLO configuration function IDs'''
    reset_periph         = 0
    enable_pkt           = 1
    set_timeout          = 2
    open_port            = 3
    open_con             = 4
    set_stack_type       = 5
    set_max_segment_size = 6

@unique
class ACCLReduceFunctions(IntEnum):
    '''CCLO reduction functions'''
    #: Elementwise sum of vectors
    SUM = 0

@unique
class ACCLCompressionFlags(IntEnum):
    '''Compression flags'''
    #: No compression
    NO_COMPRESSION = 0
    #: First input buffer is compressed
    OP0_COMPRESSED = 1
    #: Second input buffer is compressed
    OP1_COMPRESSED = 2
    #: Result buffer is compressed
    RES_COMPRESSED = 4
    #: Apply over-the-wire compression
    ETH_COMPRESSED = 8

@unique
class ACCLStreamFlags(IntEnum):
    '''Stream flags'''
    #: No streaming. All operands and results are in memory
    NO_STREAM = 0
    #: The first operand is pulled from stream instead of memory
    OP0_STREAM = 1
    #: The result is pushed to stream instead of memory
    RES_STREAM = 2

@unique
class ErrorCode(IntEnum):
    '''Error codes returned by CCLO kernel in FPGA'''
    DMA_MISMATCH_ERROR                          = (1<< 0)
    DMA_INTERNAL_ERROR                          = (1<< 1)
    DMA_DECODE_ERROR                            = (1<< 2)
    DMA_SLAVE_ERROR                             = (1<< 3)
    DMA_NOT_OKAY_ERROR                          = (1<< 4)
    DMA_NOT_END_OF_PACKET_ERROR                 = (1<< 5)
    DMA_NOT_EXPECTED_BTT_ERROR                  = (1<< 6)
    DMA_TIMEOUT_ERROR                           = (1<< 7)
    CONFIG_SWITCH_ERROR                         = (1<< 8)
    DEQUEUE_BUFFER_TIMEOUT_ERROR                = (1<< 9)
    DEQUEUE_BUFFER_SPARE_BUFFER_STATUS_ERROR    = (1<<10)
    RECEIVE_TIMEOUT_ERROR                       = (1<<11)
    DEQUEUE_BUFFER_SPARE_BUFFER_DMATAG_MISMATCH = (1<<12)
    DEQUEUE_BUFFER_SPARE_BUFFER_INDEX_ERROR     = (1<<13)
    COLLECTIVE_NOT_IMPLEMENTED                  = (1<<14)
    RECEIVE_OFFCHIP_SPARE_BUFF_ID_NOT_VALID     = (1<<15)
    OPEN_PORT_NOT_SUCCEEDED                     = (1<<16)
    OPEN_CON_NOT_SUCCEEDED                      = (1<<17)
    DMA_SIZE_ERROR                              = (1<<18)
    ARITH_ERROR                                 = (1<<19)
    PACK_TIMEOUT_STS_ERROR                      = (1<<20)
    PACK_SEQ_NUMBER_ERROR                       = (1<<21)
    COMPRESSION_ERROR                           = (1<<22)
    KRNL_TIMEOUT_STS_ERROR                      = (1<<23)
    KRNL_STS_COUNT_ERROR                        = (1<<24)
    SEGMENTER_EXPECTED_BTT_ERROR                = (1<<25)
    DMA_TAG_MISMATCH_ERROR                      = (1<<26)
