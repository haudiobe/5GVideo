import re
import subprocess
import json
import hashlib

from pathlib import Path
from enum import Enum


class ColorPrimaries(Enum):
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
    NONE = "0"
    BT709 = "1"
    BT2020_SDR = "14"
    BT2020_HLG = "18"
    BT2020_PQ = "16"


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
    while True:
        out = proc.stdout.readline().decode('utf-8')
        logfile.write(out)
        print(out.rstrip('\n'))
        if proc.poll() is not None:
            proc.stdout.flush()
            while True:
                l = proc.stdout.readline()
                if l:
                    out = l.decode('utf-8')
                    logfile.write(out)
                else:
                    break
            # print(out.rstrip('\n'))
            print(f'\n# exit code: {proc.returncode} - {cmd[0]}')
            print("#" * 128 + "\n")
            if proc.returncode != 0:
                proc.terminate()
                raise Exception(f'command failled {proc.returncode}: {out}')
            break
    logfile.flush()
    logfile.close()
    proc.stdout.close()
    proc.terminate()


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
        self.colour_primaries = from_enum(ColorPrimaries, properties.get('colourPrimaries', None))
        self.transfer_characteristics = from_enum(TransferFunction, properties.get('transferCharacteristics', None))
        self.matrix_coefficients = from_enum(MatrixCoefficients, properties.get('matrixCoefficients', None))
        self.sar = properties.get('sampleAspectRatio', None)

        self.video_full_range = properties.get('videoFullRangeFlag', None)
        self.chroma_sample_loc_type = properties.get('chromaSampleLocType', None)

        self.hdr_master_display = properties.get('HDRmasterDisplay', None)
        self.hdr_max_cll = properties.get('HDRmaxCLL', None)
        self.hdr_max_fall = properties.get('HDRmaxFALL', None)

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
            "sampleAspectRatio": self.sar,
            "videoFullRangeFlag": self.video_full_range,
            "chromaSampleLocType": self.chroma_sample_loc_type,
            "HDRmasterDisplay": self.hdr_master_display,
            "HDRmaxCLL": self.hdr_max_cll,
            "HDRmaxFALL": self.hdr_max_fall
        }


class VideoSequence(VideoInfo):

    def __init__(self, filename: str, sequence: dict = {}, contact: dict = {}, copyright: str = '', **properties):
        self.path = Path(filename).resolve()
        self.contact = contact
        self.copyright = copyright
        self.sequence = sequence
        super().__init__(**properties)

    def dump(self, path: Path):
        ref = self.sequence
        ref["URI"] = str(self.path.name)
        data = {
            "Sequence": ref,
            "Properties": self.properties,
            "copyRight": self.copyright,
            "Contact": self.contact
        }
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with open(path, 'w') as writer:
            json.dump(data, writer, indent=4)

    @staticmethod
    def from_sidecar_metadata(metadata: Path) -> 'VideoSequence':
        """minimal metadata json:
        {
            "Sequence":{
                "URI": "path/to/sequence.yuv",
                "md5": "74f80ff8c2237157060ee05f8358d88d"
            },
            "Properties": {
                "width": 4096,
                "height": 2160,
                "format": "yuv",
                "packing": "planar",
                "scan": "progressive",
                "subsampling": "420",
                "bitDepth": 8,
                "frameRate": 60.0,
                "colourPrimaries": "1",
                "transferCharacteristics": "1",
                "matrixCoefficients": "1",
                "sampleAspectRatio": "1",
                "duration": 10.0,
                "frameCount": 600,
                "startFrame": 1,
                "videoFullRangeFlag": "0",
                "chromaSampleLocType": "0"
            }
        }
        """
        data = None

        try:
            with open(metadata, 'r') as reader:
                data = json.load(reader)
        except FileNotFoundError:
            raise Exception(f'missing sidecar metadata for {metadata}')

        assert 'Sequence' in data, f"{metadata}\n'Sequence' not specified in metadata"
        assert 'URI' in data['Sequence'], f"{metadata}\n'URI' key missing from 'Sequence' metadata"
        uri = Path(data['Sequence']['URI'])

        if 'md5' not in data['Sequence']:
            print(f"/!\\ {metadata}\n'md5' key missing from 'Sequence' metadata: {uri.name}")

        data['Sequence']['Key'] = None  # The sequence key in the json files should not be used. If it exists, it may be invalid. Key is defined in the csv list.
        raw_sequence = Path(metadata).parent / Path(data['Sequence']['URI'])  # if URI is absolute, it is interpreted as such, otherwise it is interpreted relative to the metatada directory
        contact = None
        cc = None

        assert 'Properties' in data, f"{metadata}\n'Properties' missing from metadata"
        props = data['Properties']
        contact = data.get('Contact', None)
        cc = data.get('copyRight', None)

        return VideoSequence(raw_sequence, copyright=cc, contact=contact, sequence=data['Sequence'], **props)
