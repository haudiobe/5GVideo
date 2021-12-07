import subprocess
import hashlib
from pathlib import Path
from enum import Enum


def from_enum(cls: Enum, value):
    for m in cls.__members__.values():
        if m.value == value:
            return m


def md5_checksum(p: Path):
    md5 = hashlib.md5()
    block_size = 128 * md5.block_size
    with open(p, 'rb') as f:
        chunk = f.read(block_size)
        while chunk:
            md5.update(chunk)
            chunk = f.read(block_size)
        return md5.hexdigest()


def run_process(log: str, *cmd, dry_run=False, verbose=True, cwd=None):

    print("\n" + "#" * 128 + "\n")
    
    if verbose:
        print(" ".join(cmd))
    if dry_run:
        return
    logfile = open(log, 'w')
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
    while proc.poll() is None:
        out = proc.stdout.readline().decode('utf-8')
        logfile.write(out)
        print(out.rstrip('\n'))
    proc.stdout.flush()
    for l in proc.stdout:
        out = l.decode('utf-8')
        logfile.write(out)
        print(out.rstrip('\n'))
    logfile.flush()
    logfile.close()
    proc.stdout.close()
    proc.terminate()
    if proc.returncode != 0:
        raise Exception(f'exit code: {proc.returncode} - {cmd}')

