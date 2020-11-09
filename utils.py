import os
import subprocess
import json
from pathlib import Path
from enum import Enum
from typing import List
from fractions import Fraction

def run_process(log:str, *cmd):
    with open(log, 'w') as logfile:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
            while True:
                out = proc.stdout.readline().decode('utf-8')
                logfile.write(out)
                print(out.rstrip('\n'))
                if proc.poll() != None:
                    print(f'exit code: {proc.returncode} - {cmd[0]}')
                    if proc.returncode != 0:
                        raise Exception(f'command failled {proc.returncode}')
                    return

class VideoInfo:

    def __init__(self, width:int=None, height:int=None, chroma_format:str=None, chroma_subsampling:str=None, bitdepth:int=None, bitdepth_chroma:int=None, fps:int=None, framecount:int=None, color_space:str="", transfer:dict=None):
        self.width = width
        self.height = height
        self.chroma_format = chroma_format
        self.chroma_subsampling = chroma_subsampling
        self.bitdepth = bitdepth
        self.bitdepth_chroma = bitdepth if bitdepth_chroma is None else bitdepth_chroma
        self.fps = fps
        self.framecount = framecount

    @property
    def duration(self):
        return self.framecount * self.fps

    @property
    def pixfmt(self):
        return f'{self.chroma_format}{self.chroma_subsampling}'

    def to_json_dict(self):
        return self.__dict__


class VideoSequence(VideoInfo):
    
    def __init__(self, filename:str, **video_info):
        self.path = Path(filename).resolve()
        super().__init__(**video_info)
    
    @property
    def metadata_dict(self):
        md = super().to_json_dict().copy()
        del md['path']
        return md

    @staticmethod
    def with_sidecar_metadata(raw_sequence:Path):
        sidecar = raw_sequence.parent / f'{raw_sequence.stem}.json'
        try:
            with open(sidecar, 'r') as reader:
                meta = json.load(reader)
                return VideoSequence(raw_sequence, **meta)
        except FileNotFoundError:
            raise Exception(f'missing sidecar metadata for {raw_sequence}')
