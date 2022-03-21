from asyncio.log import logger
import json
from enum import Enum
from pathlib import Path

import os
from utils import from_enum
import copy

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


class VideoInfo:

    def __init__(self, **properties):
        self.width = properties.get('width', None)
        self.height = properties.get('height', None)
        self.chroma_format = from_enum(ChromaFormat, properties.get('format', None))
        self.chroma_subsampling = from_enum(ChromaSubsampling, properties.get('subsampling', None))
        self.bit_depth = properties.get('bitDepth', None)
        self.frame_rate = properties.get('frameRate', None)
        self.start_frame = properties.get('startFrame', None)
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

        self._frame_count = properties.get('frameCount', None)


    @property
    def frame_count(self):
        if bool(int(os.getenv('VCC_TEST_SINGLE_FRAME', 0))):
            return 1
        return self._frame_count

    @frame_count.setter
    def frame_count(self, count):
        self._frame_count = count

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

    @property
    def uri(self) -> str:
        return self.sequence.get('URI', None)

    def dump(self, path: Path):
        self.sequence['URI'] = str(self.path)
        data = {
            "Sequence": self.sequence,
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
                "URI": "https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/ReferenceSequences/[Location]/[Name].yuv",
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
        assert 'URI' in data['Sequence'], f"{metadata}\n'URI' missing from 'Sequence' metadata"
        
        uri = Path(data['Sequence']['URI'])
        local_file = Path(metadata).parent / uri.name  # if URI is absolute, it is interpreted as such, otherwise it is interpreted relative to the metatada directory
        if not local_file.exists():
            logger.warn(f"VideoSequence file not found: {local_file}")
        if 'md5' not in data['Sequence']:
            print(f"/!\\ {metadata}\n'md5' key missing from 'Sequence' metadata: {uri.name}")
        data['Sequence']['Key'] = None  # The sequence key in the json files should not be used. If it exists, it may be invalid. Key is defined in the csv list.
        contact = None
        cc = None

        assert 'Properties' in data, f"{metadata}\n'Properties' missing from metadata"
        props = data['Properties']
        contact = data.get('Contact', None)
        cc = data.get('copyRight', None)

        return VideoSequence(local_file, copyright=cc, contact=contact, sequence=data['Sequence'], **props)

# return a modified VideoSequence obj pointing to a converted sequence

def conversion_path(sequence: Path, suffix: str) -> Path:
    return sequence.parent / 'tmp' / f'{sequence.stem}{suffix}'

def as_10bit_sequence(yuv_in: VideoSequence) -> VideoSequence:
    yuv_out = copy.deepcopy(yuv_in)
    yuv_out.bit_depth = 10
    yuv_out.path = conversion_path(yuv_out.path, '.10bit.yuv')
    return yuv_out

def as_8bit_sequence(yuv_in: VideoSequence) -> VideoSequence:
    yuv_out = copy.deepcopy(yuv_in)
    yuv_out.bit_depth = 8
    yuv_out.path = conversion_path(yuv_out.path, '.8bit.yuv')
    return yuv_out

def as_exr2020_sequence(yuv_in: VideoSequence) -> VideoSequence:
    exr_out = copy.deepcopy(yuv_in)
    exr_out.bit_depth = 10
    exr_out.chroma_subsampling = ChromaSubsampling.CS_444
    exr_out.path = conversion_path(exr_out.path, '_2020_444_%05d.exr')
    exr_out.chroma_subsampling = ChromaSubsampling.CS_444
    exr_out.chroma_format = ChromaFormat.RGB
    exr_out.transfer_characteristics = TransferFunction.NONE
    exr_out.colour_primaries = ColorPrimaries.BT_2020
    exr_out.video_full_range = 1
    return exr_out

