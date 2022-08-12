=============
API Reference
=============

.. autosummary::
    :toctree: generated

ACCL
====

.. automodule:: pyaccl.accl
    :members:
    :undoc-members:
    :exclude-members: ACCLCommunicator, addr, write, use_tcp, use_udp, dummy_address, get_hwid, get_retcode, self_check_return_value, prepare_call, check_return_value, call_sync, call_async, open_con, open_port, setup_rx_buffers, configure_arithmetic, configure_communicator, dump_exchange_memory, dump_rx_buffers, init_connection, set_max_segment_size, set_timeout
    :show-inheritance:

Buffer
======

``ACCLBuffer`` objects are similar to Pynq buffers but may also interact 
with the ACCL emulator rather than just Alveo memory. Users should not
create ACCL buffers explicitly but should instead utilize the ``allocate()``
function of a specific ACCL instance.

.. automodule:: pyaccl.buffer
    :members:
    :undoc-members:
    :exclude-members: next_free_address, preallocate_memory
    :show-inheritance:

Constants
=========

.. automodule:: pyaccl.constants
    :members:
    :undoc-members:
    :exclude-members: CCLOCfgFunc, CCLOp
    :show-inheritance:
