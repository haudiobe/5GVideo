import os
import subprocess
import json
from pathlib import Path
from enum import Enum, IntEnum
from typing import List
from fractions import Fraction

class ColourPrimaries(Enum):
    BT_709 = "1"
    BT_2020 = "9"

class MatrixCoefficients(Enum):
    BT_709 = "1"
    BT_2020 = "9"

class ChromaFormat(Enum):
    YUV = 'yuv'
    RGB = 'rgb'

class ChromaSubsampling(Enum):
    CS_400 = '400'
    CS_420 = '420'
    CS_422 = '422'
    CS_444 = '444'

class TransferFunction(Enum):
    BT709 = "1"
    BT2020_SDR = "14"
    BT2020_HLG = "18"
    BT2020_PQ = "16"


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
        self.start_frame = properties.get('startFrame', None)
        self.frame_count = properties.get('frameCount', None)
        self.packing = properties.get('packing', None)
        self.scan = properties.get('scan', None)
        self.colour_primaries = from_enum(ColourPrimaries, properties.get('colourPrimaries', None))
        self.transfer_characteristics = from_enum(TransferFunction, properties.get('transferCharacteristics', None))
        self.matrix_coefficients = from_enum(MatrixCoefficients, properties.get('matrixCoefficients', None))
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
        assert self.colour_primaries in [ColourPrimaries.BT_709, ColourPrimaries.BT_2020], f'unsupported colour primaries, expected 1 or 9, got: {self.colour_primaries}'
        
        if self.colour_primaries == ColourPrimaries.BT_709:
            assert self.transfer_characteristics == TransferFunction.BT709, 'unsupported transfer characteristics for colour primaries 1'
            assert self.matrix_coefficients == MatrixCoefficients.BT_709, 'unsupported matrix coefficient for colour primaries 1'
        elif self.colour_primaries == ColourPrimaries.BT_2020:
            assert self.matrix_coefficients == MatrixCoefficients.BT_2020, 'unsupported matrix coefficient for colour primaries 9'
            assert self.transfer_characteristics in [TransferFunction.BT2020_SDR, TransferFunction.BT2020_HLG, TransferFunction.BT2020_HLG], 'unsupported transfer characteristics for colour primaries 9'

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
            "startFrame": self.start_frame,
            "frameRate": self.frame_rate,
            "frameCount": self.frame_count,
            "format": self.chroma_format.value,
            "packing": self.packing,
            "scan": self.scan,
            "subsampling": self.chroma_subsampling.value,
            "bitDepth": self.bit_depth,
            "colourPrimaries": self.colour_primaries.value,
            "transferCharacteristics": self.transfer_characteristics.value,
            "matrixCoefficients": self.matrix_coefficients.value,
            "sampleAspectRatio": self.sar
        }


class VideoSequence(VideoInfo):
    
    def __init__(self, filename:str, contact:dict = None, copyright:str = None, **properties):
        self.path = Path(filename).resolve()
        self.contact = contact
        self.copyright = copyright
        super().__init__(**properties)

    @staticmethod
    def from_sidecar_metadata(metadata:Path):
        # https://github.com/haudiobe/5G-Video-Content/blob/main/3gpp-raw-schema.json
        try:
            with open(metadata, 'r') as reader:
                data = json.load(reader)
                assert 'Sequence' in data, f"{metadata}\n'Sequence' not specified in metadata"
                assert 'URI' in data['Sequence'], f"{metadata}\n'URI' key missing from 'Sequence' metadata"
                assert 'Key' in data['Sequence'], f"{metadata}\n'Key' key missing from 'Sequence' metadata"
                assert 'Name' in data['Sequence'], f"{metadata}\n'Name' key missing from 'Sequence' metadata"
                assert 'md5' in data['Sequence'], f"{metadata}\n'md5' key missing from 'Sequence' metadata"
                # assert 'size' in data['Sequence'], f"{metadata}\n'size' key missing from 'Sequence' metadata"
                assert 'Background' in data['Sequence'], f"{metadata}\n'Background' key missing from 'Sequence' metadata"
                assert 'Scenario' in data['Sequence'], f"{metadata}\n'Scenario' key missing from 'Sequence' metadata"
                assert 'thumbnail' in data['Sequence'], f"{metadata}\n'thumbnail' key missing from 'Sequence' metadata"
                assert 'preview' in data['Sequence'], f"{metadata}\n'preview' key missing from 'Sequence' metadata"
                assert 'TR26.955' in data['Sequence'], f"{metadata}\n'TR26.955' key missing from 'Sequence' metadata"
                # if URI is absolute, it is interpreted as such, otherwise it is interpreted relative to the metatada directory
                raw_sequence = Path(metadata).parent / Path(data['Sequence']['URI'])
                assert ('Properties' in data) and type(data['Properties']) == dict, 'invalid sequence description'
                props = data['Properties']
                contact = data.get('Contact', None )
                cc = data.get('copyRight', None )
                return VideoSequence(raw_sequence, copyright=cc, contact=contact, **props)
        except FileNotFoundError:
            raise Exception(f'missing sidecar metadata for {raw_sequence}')
            
