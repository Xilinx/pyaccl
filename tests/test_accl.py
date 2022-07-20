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

import numpy as np
from pyaccl import accl, ACCLReduceFunctions, ACCLStreamFlags, ACCLBuffer
import pytest
import os
from mpi4py import MPI

START_PORT = os.getenv("ACCL_SIM_START_PORT")
if START_PORT is None:
    START_PORT = 5500

allocated_buffers = {}

def get_buffers(count, op0_dt, op1_dt, res_dt):
    #retrieve buffers of right datatype
    op0_buf = allocated_buffers[op0_dt][0]
    op1_buf = allocated_buffers[op1_dt][1]
    res_buf = allocated_buffers[res_dt][2]
    #trim to correct length
    op0_buf = op0_buf[0:count]
    op1_buf = op1_buf[0:count]
    res_buf = res_buf[0:count]
    #fill with random data
    op0_buf[:] = np.random.randn(count).astype(op0_dt)
    op1_buf[:] = np.random.randn(count).astype(op1_dt)
    res_buf[:] = np.random.randn(count).astype(res_dt)
    return op0_buf, op1_buf, res_buf

@pytest.fixture(scope="module", params=[{"rxbuf_size":16*1024, "tcp":True, "hw":False}])
def cclo_inst(request):
    # get communicator size and our local rank in it
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    local_rank = comm.Get_rank()

    #set a random seed to make it reproducible
    np.random.seed(42+local_rank)

    ranks = []
    for i in range(world_size):
        ranks.append({"ip": "127.0.0.1", "port": START_PORT+world_size+i, "session_id":i, "max_segment_size": request.param["rxbuf_size"]})

    #configure FPGA and CCLO cores with the default 16 RX buffers of size given by request.param["rxbuf_size"]
    cclo_ret = accl(    ranks,
                        local_rank,
                        bufsize=request.param["rxbuf_size"],
                        protocol=("TCP" if request.param["tcp"] else "UDP"),
                        sim_sock="tcp://localhost:"+str(START_PORT+local_rank)
                    )
    cclo_ret.set_timeout(10**8)

    #allocate 3 buffers of max size used in our tests (128 elements),
    #for each supported datatype
    count = world_size*128
    for dt in [np.float16, np.float32, np.float64, np.int32, np.int64]:
        allocated_buffers[dt] = []
        for i in range(3):
            buf = cclo_ret.allocate((count,), dtype=dt)
            allocated_buffers[dt].append(buf)

    #barrier here to make sure all the devices are configured before testing
    comm.barrier()

    # pass CCLO instance to test code
    yield cclo_ret

    # teardown
    cclo_ret.deinit()

@pytest.fixture
def world_size(cclo_inst):
    return len(cclo_inst.communicators[0].ranks)

@pytest.fixture
def local_rank(cclo_inst):
    return cclo_inst.communicators[0].local_rank

@pytest.fixture(autouse = True)
def intertest_barrier():
    comm = MPI.COMM_WORLD
    comm.barrier()
    yield
    comm.barrier()

@pytest.mark.parametrize("count", [1, 16, 21, 128])
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.int32, np.int64])
def test_copy(cclo_inst, count, dt):
    op_buf, _, res_buf = get_buffers(count, dt, dt, dt)
    cclo_inst.copy(op_buf, res_buf, count)
    assert np.isclose(op_buf, res_buf, atol=1e-02).all()

@pytest.mark.parametrize("count", [1, 16, 21, 128])
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.int32, np.int64])
def test_combine(cclo_inst, count, dt):
    op0_buf, op1_buf, res_buf = get_buffers(count, dt, dt, dt)
    cclo_inst.combine(count, ACCLReduceFunctions.SUM, op0_buf, op1_buf, res_buf)
    assert np.isclose(op0_buf+op1_buf, res_buf, atol=1e-02).all()

@pytest.mark.mpi
@pytest.mark.parametrize("count", [1, 16, 21, 128])
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.int32, np.int64])
def test_sendrecv(cclo_inst, world_size, local_rank, count, dt):
    op_buf, _, res_buf = get_buffers(count, dt, dt, dt)
    # send to next rank; receive from previous rank; send back data to previous rank; receive from next rank; compare
    next_rank = (local_rank+1)%world_size
    prev_rank = (local_rank+world_size-1)%world_size
    cclo_inst.send(op_buf, count, next_rank, tag=0)
    cclo_inst.recv(res_buf, count, prev_rank, tag=0)
    cclo_inst.send(res_buf, count, prev_rank, tag=1)
    cclo_inst.recv(res_buf, count, next_rank, tag=1)
    assert np.isclose(op_buf, res_buf).all()

@pytest.mark.mpi
@pytest.mark.parametrize("count", [1, 16, 21, 128])
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.int32, np.int64])
def test_sendrecv_strm(cclo_inst, world_size, local_rank, count, dt):
    #NOTE: this requires loopback on the external stream interface
    op_buf, _, res_buf = get_buffers(count, dt, dt, dt)
    # send to next rank; receive from previous rank; send back data to previous rank; receive from next rank; compare
    next_rank = (local_rank+1)%world_size
    prev_rank = (local_rank+world_size-1)%world_size
    cclo_inst.send(op_buf, count, next_rank, stream_flags=ACCLStreamFlags.RES_STREAM, tag=0)
    # recv is direct, no call required
    cclo_inst.send(res_buf, count, prev_rank, stream_flags=ACCLStreamFlags.OP0_STREAM, tag=5)
    cclo_inst.recv(res_buf, count, next_rank, tag=5)
    assert np.isclose(op_buf, res_buf).all()

@pytest.mark.mpi
@pytest.mark.parametrize("count", [1, 16, 21, 128])
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.int32, np.int64])
def test_sendrecv_fanin(cclo_inst, world_size, local_rank, count, dt):
    if cclo_inst.protocol == "UDP":
        pytest.skip("No RX fanin support for UDP")
    op_buf, _, res_buf = get_buffers(count, dt, dt, dt)
    # send to next rank; receive from previous rank; send back data to previous rank; receive from next rank; compare
    if local_rank != 0:
        for i in range(len(op_buf)):
            op_buf[i] = i+local_rank
        cclo_inst.send(op_buf, count, 0, tag=0)
    else:
        for i in range(world_size):
            if i == local_rank:
                continue
            cclo_inst.recv(res_buf, count, i, tag=0)
            for j in range(len(op_buf)):
                op_buf[j] = j+i
            assert np.isclose(op_buf, res_buf).all()

@pytest.mark.mpi
def test_barrier(cclo_inst):
    cclo_inst.barrier()

@pytest.mark.mpi
@pytest.mark.parametrize("root", [0, 1])
@pytest.mark.parametrize("count", [1, 16, 21, 128])
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.int32, np.int64])
def test_bcast(cclo_inst, local_rank, root, count, dt):
    op_buf, _, res_buf = get_buffers(count, dt, dt, dt)
    op_buf[:] = [42+i for i in range(len(op_buf))]
    cclo_inst.bcast(op_buf if root == local_rank else res_buf, count, root=root)
    if local_rank != root:
        assert np.isclose(op_buf, res_buf).all()

@pytest.mark.mpi
@pytest.mark.parametrize("root", [0, 1])
@pytest.mark.parametrize("count", [1, 16, 21, 128])
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.int32, np.int64])
def test_scatter(cclo_inst, world_size, local_rank, root, count, dt):
    op_buf, _, res_buf = get_buffers(count*world_size, dt, dt, dt)
    op_buf[:] = [1.0*i for i in range(op_buf.size)]
    cclo_inst.scatter(op_buf, res_buf, count, root=root)
    assert np.isclose(op_buf[local_rank*count:(local_rank+1)*count], res_buf[0:count]).all()

@pytest.mark.mpi
@pytest.mark.parametrize("root", [0, 1])
@pytest.mark.parametrize("count", [1, 16, 21, 128])
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.int32, np.int64])
def test_gather(cclo_inst, world_size, local_rank, root, count, dt):
    op_buf, _, res_buf = get_buffers(count*world_size, dt, dt, dt)
    op_buf[:] = [1.0*(local_rank+i) for i in range(op_buf.size)]
    cclo_inst.gather(op_buf, res_buf, count, root=root)
    if local_rank == root:
        for i in range(world_size):
            assert np.isclose(res_buf[i*count:(i+1)*count], [1.0*(i+j) for j in range(count)]).all()

@pytest.mark.mpi
@pytest.mark.parametrize("count", [1, 16, 21, 128])
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.int32, np.int64])
def test_allgather(cclo_inst, world_size, local_rank, count, dt):
    op_buf, _, res_buf = get_buffers(count*world_size, dt, dt, dt)
    op_buf[:] = [1.0*(local_rank+i) for i in range(op_buf.size)]
    cclo_inst.allgather(op_buf, res_buf, count)
    for i in range(world_size):
        assert np.isclose(res_buf[i*count:(i+1)*count], [1.0*(i+j) for j in range(count)]).all()

@pytest.mark.mpi
@pytest.mark.parametrize("root", [0, 1])
@pytest.mark.parametrize("func", [ACCLReduceFunctions.SUM])
@pytest.mark.parametrize("count", [1, 16, 21, 128])
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.int32, np.int64])
def test_reduce(cclo_inst, world_size, local_rank, root, count, func, dt):
    op_buf, _, res_buf = get_buffers(count, dt, dt, dt)
    op_buf[:] = [1.0*i*(local_rank+1) for i in range(op_buf.size)]
    cclo_inst.reduce(op_buf, res_buf, count, root, func)
    if local_rank == root:
        assert np.isclose(res_buf, sum(range(world_size+1))*op_buf).all()

@pytest.mark.mpi
@pytest.mark.parametrize("func", [ACCLReduceFunctions.SUM])
@pytest.mark.parametrize("count", [1, 16, 21, 128])
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.int32, np.int64])
def test_reduce_scatter(cclo_inst, world_size, local_rank, count, func, dt):
    op_buf, _, res_buf = get_buffers(world_size*count, dt, dt, dt)
    op_buf[:] = [1.0*i for i in range(op_buf.size)]
    cclo_inst.reduce_scatter(op_buf, res_buf, count, func)
    full_reduce_result = world_size*op_buf
    assert np.isclose(res_buf[0:count], full_reduce_result[local_rank*count:(local_rank+1)*count]).all()

@pytest.mark.mpi
@pytest.mark.parametrize("func", [ACCLReduceFunctions.SUM])
@pytest.mark.parametrize("count", [16, 21, 128])
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.int32, np.int64])
def test_allreduce(cclo_inst, world_size, count, func, dt):
    if count < world_size:
        pytest.skip("Too few array elements for ring-allreduce test")
    op_buf, _, res_buf = get_buffers(count, dt, dt, dt)
    op_buf[:] = [1.0*i for i in range(op_buf.size)]
    cclo_inst.allreduce(op_buf, res_buf, count, func)
    full_reduce_result = world_size*op_buf
    assert np.isclose(res_buf, full_reduce_result).all()

@pytest.mark.mpi
def test_multicomm(cclo_inst, world_size, local_rank):
    if world_size < 4:
        pytest.skip("Too few ranks for multi-communicator test")
    if local_rank not in [0, 2, 3]:
        return
    cclo_inst.split_communicator([0, 2, 3])
    used_comm = len(cclo_inst.communicators) - 1
    count = 128
    op_buf, _, res_buf = get_buffers(count, np.float32, np.float32, np.float32)
    # start with a send/recv between ranks 0 and 2 (0 and 1 in the new communicator)
    if local_rank == 0:
        cclo_inst.send(op_buf, count, 1, tag=0, comm_id=used_comm)
        cclo_inst.recv(res_buf, count, 1, tag=1, comm_id=used_comm)
        assert np.isclose(res_buf, op_buf).all()
    elif local_rank == 2:
        cclo_inst.recv(res_buf, count, 0, tag=0, comm_id=used_comm)
        cclo_inst.send(res_buf, count, 0, tag=1, comm_id=used_comm)
    # do an all-reduce on the new communicator
    op_buf[:] = [1.0*i for i in range(op_buf.size)]
    cclo_inst.allreduce(op_buf, res_buf, count, 0, comm_id=used_comm)
    expected_allreduce_result = 3*op_buf
    assert np.isclose(res_buf, expected_allreduce_result).all()

@pytest.mark.mpi
def test_multicomm_allgather(cclo_inst, world_size, local_rank):
    if local_rank < world_size // 2:
        new_group = [i for i in range(world_size // 2)]
    else:
        new_group = [i for i in range(world_size // 2, world_size, 1)]
    cclo_inst.split_communicator(new_group)
    used_comm = len(cclo_inst.communicators) - 1
    count = 128
    op_buf, _, res_buf = get_buffers(count * world_size, np.float32, np.float32, np.float32)
    op_buf[:] = [1.0*(local_rank+i) for i in range(op_buf.size)]

    cclo_inst.allgather(op_buf, res_buf, count, comm_id=used_comm)

    for i, g in enumerate(new_group):
        assert np.isclose(res_buf[i*count:(i+1)*count], [1.0*(g+j) for j in range(count)]).all()

    op_buf[:] = [1.0 * local_rank for i in range(op_buf.size)]
    cclo_inst.bcast(op_buf, count, root=0)
    if local_rank != 0:
        assert np.isclose(op_buf, [0.0 for i in range(op_buf.size)]).all()

@pytest.mark.mpi
@pytest.mark.parametrize("count", [16, 21, 128])
def test_eth_compression(cclo_inst, world_size, local_rank, count):
    op_buf, _, res_buf = get_buffers(count, np.float32, np.float32, np.float32)
    # paint source buffer
    op_buf[:] = [3.14159*i for i in range(op_buf.size)]

    # send data, with compression, from 0 to 1
    if local_rank == 0:
        cclo_inst.send(op_buf, count, 1, tag=0, compress_dtype=np.dtype('float16'))
    elif local_rank == 1:
        cclo_inst.recv(res_buf, count, 0, tag=0, compress_dtype=np.dtype('float16'))
        # check data; since there seems to be some difference between
        # numpy and FPGA fp32 <-> fp16 conversion, allow 1% relative error, and 0.01 absolute error
        assert np.isclose(op_buf.astype(np.float16).astype(np.float32), res_buf, rtol=1e-02, atol=1e-02).all()

    cclo_inst.bcast(op_buf if local_rank == 0 else res_buf, count, root=0, compress_dtype=np.dtype('float16'))
    if local_rank > 0:
        assert np.isclose(op_buf.astype(np.float16).astype(np.float32), res_buf, rtol=1e-02, atol=1e-02).all()

    cclo_inst.reduce(op_buf, res_buf, count, 0, ACCLReduceFunctions.SUM, compress_dtype=np.dtype('float16'))
    if local_rank == 0:
        assert np.isclose(res_buf, world_size*op_buf, rtol=1e-02, atol=1e-02).all()

    cclo_inst.allreduce(op_buf, res_buf, count, ACCLReduceFunctions.SUM, compress_dtype=np.dtype('float16'))
    assert np.isclose(res_buf, world_size*op_buf, rtol=1e-02, atol=1e-02).all()

    # re-generate buffers for asymmetric size collectives - (reduce-)scatter, (all-)gather
    op_buf, _, res_buf = get_buffers(count*world_size, np.float32, np.float32, np.float32)
    # paint source buffer
    op_buf[:] = [3.14159*i for i in range(op_buf.size)]

    cclo_inst.scatter(op_buf, res_buf, count, 0, compress_dtype=np.dtype('float16'))
    if local_rank > 0:
        assert np.isclose(op_buf[local_rank*count:(local_rank+1)*count].astype(np.float16).astype(np.float32), res_buf[0:count], rtol=1e-02, atol=1e-02).all()

    cclo_inst.gather(op_buf, res_buf, count, 0, compress_dtype=np.dtype('float16'))
    if local_rank == 0:
        for i in range(world_size):
            assert np.isclose(op_buf[0:count].astype(np.float16).astype(np.float32), res_buf[i*count:(i+1)*count], rtol=1e-02, atol=1e-02).all()

    cclo_inst.allgather(op_buf[local_rank*count:(local_rank+1)*count], res_buf, count, compress_dtype=np.dtype('float16'))
    assert np.isclose(op_buf.astype(np.float16).astype(np.float32), res_buf, rtol=1e-02, atol=1e-02).all()

    cclo_inst.reduce_scatter(op_buf, res_buf, count, ACCLReduceFunctions.SUM, compress_dtype=np.dtype('float16'))
    expected_res = world_size*op_buf[local_rank*count:(local_rank+1)*count]
    assert np.isclose(expected_res.astype(np.float16).astype(np.float32), res_buf[0:count], rtol=1e-02, atol=1e-02).all()

