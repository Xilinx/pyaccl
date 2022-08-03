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

import os
import subprocess
import argparse
import signal

def emulate_cclo_instances(ranks: int, log_level: int, start_port: int, use_udp: bool, kernel_loopback: bool):
    env = os.environ.copy()
    print("Starting emulator...")
    env['LOG_LEVEL'] = str(log_level)
    args = ['mpirun', '-np', str(ranks), '--tag-output', 'cclo_emu', 'udp' if use_udp else 'tcp',
            str(start_port), "loopback" if kernel_loopback else ""]
    print(' '.join(args))
    with subprocess.Popen(args, env=env, stderr=subprocess.DEVNULL) as p:
        try:
            p.wait()
        except KeyboardInterrupt:
            try:
                print("Stopping emulator...")
                p.send_signal(signal.SIGINT)
                p.wait()
            except KeyboardInterrupt:
                try:
                    print("Force stopping emulator...")
                    p.kill()
                    p.wait()
                except KeyboardInterrupt:
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                    print("Terminating emulator...")
                    p.terminate()
                    p.wait()
        if p.returncode != 0:
            print(f"Emulator exited with error code {p.returncode}")

def run_emulator():
    parser = argparse.ArgumentParser(description='Run ACCL emulator')
    parser.add_argument('-n', '--nranks', type=int, default=1,
                        help='How many ranks to use for the emulator')
    parser.add_argument('-l', '--log-level', type=int, default=3,
                        help='Log level to use, defaults to 3 (info)')
    parser.add_argument('-s', '--start-port', type=int, default=5500,
                        help='Start port of emulator')
    parser.add_argument('-u', '--udp', action='store_true', default=False,
                        help='Run emulator over UDP instead of TCP')
    parser.add_argument('--no-kernel-loopback', action='store_true', default=False,
                        help="Do not connect user kernel data ports in loopback")
    args = parser.parse_args()
    emulate_cclo_instances(args.nranks, args.log_level, args.start_port, args.udp,
        not args.no_kernel_loopback)

