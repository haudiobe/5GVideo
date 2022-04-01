import enum
from pathlib import Path
ENCODING = 'utf-8-sig'

BITSTREAMS_DIR = 'Bitstreams'
SEQUENCES_DIR = 'ReferenceSequences'

# scenario/codec/CFG/encoder.cfg
def encoder_cfg_path(cfg_key: str, bitstreams_dir:Path ) -> Path:
    """
    given an encoder config key, 
        returns the location of the encoder config file relative to bitstreams_dir
    """
    arr = cfg_key.split('-')
    if len(arr) == 3:
        scenario, enc, _ = arr
        if scenario == 'S1':
            scenario = 'Scenario-1-FHD'
        elif scenario == 'S2':
            scenario = 'Scenario-2-4K'
        elif scenario == 'S3':
            scenario = 'Scenario-3-Screen'
        elif scenario == 'S4':
            scenario = 'Scenario-4-Sharing'
        elif scenario == 'S5':
            scenario = 'Scenario-5-Gaming'
        else:
            return bitstreams_dir / f'{cfg_key}.cfg'
        if enc == 'HM':
            enc = '265'
        elif enc == 'SCC':
            enc = '265'
        elif enc == 'SCM':
            enc = '265'
        elif enc == 'JM':
            enc = '264'
        return bitstreams_dir / f'{scenario}/{enc}/CFG/{cfg_key}.cfg'
    return bitstreams_dir / f'{cfg_key}.cfg'


# scenario/codec/streams.csv
class AnchorList:
    KEY = '#Key'  # directory where anchor is stored
    CLAUSE = 'Clause'
    REF_SEQ = 'Reference Sequence'
    REF_ENC = 'Reference_Encoder'
    TEST_ENC = 'Test_Encoder'
    CFG = 'Configuration'
    VARIANTS = 'Variations'
    VARIANT_KEY = 'Anchor_Key'  # template for variant json filename


# scenario/reference-sequence.csv
class RefSequenceList:
    KEY = '#Key'
    NAME = 'Name'
    REF = 'Reference'
    LOC = 'Location'
    DUR = 'Duration'


class M:

    def __init__(self, csv_key: str, json_key: str = None) -> None:
        self.csv_key = csv_key
        self.json_key = json_key if json_key else csv_key

    def __repr__(self):
        return self.csv_key

    def __str__(self):
        return self.csv_key


class Metric(enum.Enum):

    @classmethod
    def from_json_key(cls, key:str):
        for m in cls.__members__.values():
            if m.json_key == key:
                return m

    @classmethod
    def from_csv_key(cls, key:str):
        for m in cls.__members__.values():
            if m.csv_key == key:
                return m

    @property
    def csv_key(self):
        return self.value.csv_key

    @property
    def json_key(self):
        return self.value.json_key

    # @property
    # def key(self):
    #     return self.value.csv_key

    PSNR = M("psnr", "PSNR")
    PSNR_Y = M("y_psnr", "YPSNR")
    PSNR_U = M("u_psnr", "UPSNR")
    PSNR_V = M("v_psnr", "VPSNR")
    MSSSIM = M("ms_ssim", "MS_SSIM")
    VMAF = M("vmaf", "VMAF")
    BITRATE = M("bitrate", "Bitrate")
    BITRATELOG = M("bitrate_log", "BitrateLog")
    ENCODETIME = M("encode_time", "EncodeTime")
    DECODETIME = M("decode_time", "DecodeTime")
    DELTAE100 = M("de100", "DE100")
    WTPSNR = M("wpsnr", "WPSNR")
    WTPSNR_Y = M("y_wpsnr", "YWPSNR")
    WTPSNR_U = M("u_wpsnr", "UWPSNR")
    WTPSNR_V = M("v_wpsnr", "VWPSNR")
    PSNR_L0100 = M("psnr100", "PSNRL100")


# Table 5.5.6-1 Result Format for SDR/HLG
SDR_METRICS = (
    Metric.BITRATE,
    Metric.PSNR,
    Metric.PSNR_Y,
    Metric.PSNR_U,
    Metric.PSNR_V,
    Metric.MSSSIM, 
    Metric.VMAF,
    Metric.BITRATELOG,
    Metric.ENCODETIME,
    Metric.DECODETIME
)


# Table 5.5.6-2 Result Format for HDR
HDR_METRICS = (
    Metric.BITRATE,
    Metric.PSNR_Y,
    Metric.PSNR_U,
    Metric.PSNR_V,
    Metric.PSNR,
    Metric.WTPSNR_Y,
    Metric.WTPSNR_U,
    Metric.WTPSNR_V,
    Metric.WTPSNR,
    Metric.PSNR_L0100,
    Metric.DELTAE100,
    Metric.BITRATELOG,
    Metric.ENCODETIME,
    Metric.DECODETIME
)

