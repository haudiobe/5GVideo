import os
import subprocess
import json
from pathlib import Path
from enum import Enum, IntEnum
from typing import List
from fractions import Fraction

class ColourPrimaries(IntEnum):
    BT_709 = 1
    BT_2020 = 9

class ChromaFormat(Enum):
    YUV = 'yuv'
    RGB = 'rgb'

class ChromaSubsampling(Enum):
    CS_400 = '400'
    CS_420 = '420'
    CS_422 = '422'
    CS_444 = '444'
    
def from_enum(cls:Enum, value):
    for m in cls.__members__.values():
        if m.value == value:
            return m

def run_process(log:str, *cmd, dry_run=False):
    if dry_run:
        print(" ".join(cmd))
        return
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

    def __init__(self, **properties):
        self.width = properties.get('width', None)
        self.height = properties.get('height', None)
        self.chroma_format = from_enum(ChromaFormat, properties.get('format', None))
        self.chroma_subsampling = from_enum(ChromaSubsampling, properties.get('subsampling', None))
        self.bit_depth = properties.get('bitDepth', None)
        self.frame_rate = properties.get('frameRate', None)
        self.frame_count = properties.get('frameCount', None)
        self.packing = properties.get('packing', None)
        self.scan = properties.get('scan', None)
        self.colour_primaries = from_enum(ColourPrimaries, properties.get('colourPrimaries', None))
        self.transfer_characteristics = properties.get('transferCharacteristics', None)
        self.matrix_coefficients = properties.get('matrixCoefficients', None)
        self.sar = properties.get('sampleAspectRatio', None)
        self.is_valid_sequence()

    def is_valid_sequence(self):
        assert self.width != None and type(self.width) == int, f'invalid width: {self.width}'
        assert self.height != None and type(self.height) == int, f'invalid height: {self.height}'
        assert self.chroma_format in [ChromaFormat.YUV, ChromaFormat.RGB], f'invalid chroma format: {self.chroma_format}'
        # should just use booleans for interleaved/interlaced
        assert self.packing in ['planar', 'interleaved'], f'invalid packing: {self.packing}'
        assert self.scan in ['progressive', 'interlaced'], f'invalid scan: {self.scan}'
        assert self.chroma_subsampling in [ChromaSubsampling.CS_400, ChromaSubsampling.CS_420, ChromaSubsampling.CS_422, ChromaSubsampling.CS_444], f'invalid subsampling: {self.chroma_subsampling}'
        assert self.bit_depth in [8, 10, 12, 16], f'invalid bitdepth: {self.bit_depth}'
        assert self.colour_primaries in [ColourPrimaries.BT_709, ColourPrimaries.BT_2020], 'unsupported colour primaries, expected 1 or 9'
        
        if self.colour_primaries == ColourPrimaries.BT_709:
            # cp=1, tc=1, mc=1 is BT.709
            assert self.transfer_characteristics == 1, 'unsupported transfer characteristics for colour primaries 1'
            assert self.matrix_coefficients == 1, 'unsupported matrix coefficient for colour primaries 1'
        elif self.colour_primaries == ColourPrimaries.BT_2020:
            # cp=9, tc=14, mc=9 is BT.2020 with SDR
            # cp=9, tc=16, mc=9 is BT.2020 with HDR PQ
            # cp=9, tc=18, mc=9 is BT.2020 with HDR HLG
            assert self.matrix_coefficients == 9, 'unsupported matrix coefficient for colour primaries 9'
            assert self.transfer_characteristics in [14, 16, 18], 'unsupported transfer characteristics for colour primaries 9'

    @property
    def interleaved(self):
        return self.packing == 'interleaved'

    @property
    def interlaced(self):
        return self.scan == 'interlaced'

    @property
    def properties(self):
        return {
            "width": self.width,
            "height": self.height,
            "frameRate": self.frame_rate,
            "frameCount": self.frame_count,
            "format": self.chroma_format.value,
            "packing": self.packing,
            "scan": self.scan,
            "subsampling": self.chroma_subsampling.value,
            "bitDepth": self.bit_depth,
            "colourPrimaries": self.colour_primaries.value,
            "transferCharacteristics": self.transfer_characteristics, 
            "matrixCoefficients": self.matrix_coefficients,
            "sampleAspectRatio": self.sar
        }


class VideoSequence(VideoInfo):
    
    def __init__(self, filename:str, **properties):
        self.path = Path(filename).resolve()
        super().__init__(**properties)

    @staticmethod
    def from_sidecar_metadata(metadata:Path):
        try:
            with open(metadata, 'r') as reader:
                sequence_metadata = json.load(reader)
                assert 'Sequence' in sequence_metadata, "Sequence not specified in metadata"
                assert 'URI' in sequence_metadata['Sequence'], "sequence URI not specified in metadata"
                # if URI is absolute, it is interpreted as such, otherwise it is interpreted relative to the metatada directory
                raw_sequence = Path(metadata).parent / Path(sequence_metadata['Sequence']['URI'])
                assert ('Properties' in sequence_metadata) and type(sequence_metadata['Properties']) == dict, 'invalid sequence description'
                props = sequence_metadata['Properties']
                return VideoSequence(raw_sequence, **props)
        except FileNotFoundError:
            raise Exception(f'missing sidecar metadata for {raw_sequence}')
            
