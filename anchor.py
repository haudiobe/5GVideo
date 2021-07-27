import json
import csv
from pathlib import Path
from typing import Dict, Generator, Tuple, Iterable

from utils import VideoSequence, md5_checksum

ENCODING = 'utf-8-sig'

# scenario/anchors.csv
class AnchorList:
    KEY = '#Key' # directory where anchor is stored
    CLAUSE = 'Clause'
    REF_SEQ = 'Reference Sequence'
    REF_ENC = 'Reference_Encoder'
    TEST_ENC = 'Test_Encoder'
    CFG = 'Configuration'
    VARIANTS = 'Variations'
    VARIANT_KEY = 'Anchor_Key' # template for variant json filename

# scenario/sequences.csv
class RefSequenceList:
    KEY = '#Key'
    NAME = 'Name'
    REF = 'Reference'
    LOC = 'Location'
    DUR = 'Duration'


class ReconstructionMeta:
    def __init__(self, decoder_id:str, reconstructed:Path, decoder_log:Path, md5=True):
        self.decoder_id = decoder_id
        self.reconstructed = reconstructed
        if md5:
            self.reconstructed_md5 = md5_checksum(self.reconstructed)
        else:
            self.reconstructed_md5 = None
        self.decoder_log = decoder_log

    def to_dict(self) -> dict:
        return {
            "decoder": self.decoder_id,
            "log-file": self.decoder_log.name,
            "md5": self.reconstructed_md5
        }

class VariantData:

    @classmethod
    def load(cls, fp:Path) -> 'VariantData':
        assert fp.exists(), f'File not found: {fp}'
        with open(fp, 'r') as fo:
            data = json.load(fo)
            generation = data.get("Generation", None)
            bitstream = data.get("Bitstream", None)
            reconstruction = data.get("Reconstruction", None)
            metrics = data.get("Metrics", None)
            if metrics != None:
                for k, v in metrics.items():
                    metrics[k] = float(v)
            verification = data.get("Verification", None)
            contact = data.get("contact", None)
            copyright = data.get("copyRight", None)
            return VariantData(generation, bitstream, reconstruction, metrics, verification, contact, copyright)

    @classmethod
    def new(cls, a:'AnchorTuple', variant_id:str, variant_cli:str, bitstream_fp:Path, encoder_log:Path) -> 'VariantData':
        assert bitstream_fp.exists(), f'{bitstream_fp} not found'
        generation = {
            "key": variant_id,
            "sequence": a.reference.path.name,
            "encoder": a.encoder_id,
            "config-file": a.encoder_cfg,
            "variant": variant_cli,
            "log-file": encoder_log.name
        }
        bitstream = {
            "key": variant_id,
            "URI": str(bitstream_fp),
            "md5": md5_checksum(bitstream_fp),
            "size": bitstream_fp.stat().st_size
        }
        contact = {
            'Company': a.reference.contact['Company'],
            'e-mail': a.reference.contact['e-mail']
        }
        copyright = a.reference.copyright
        return VariantData(generation, bitstream, None, None, None, contact, copyright)

    def __init__(self, generation:dict=None, bitstream:dict=None, reconstruction:dict=None, metrics:dict=None, verification:dict=None, contact:dict=None, copyright:str=''):
        self._generation = generation
        self._bitstream = bitstream
        self._contact = contact
        self._metrics = metrics
        self._reconstruction = reconstruction
        self._verification = verification
        self._copyright = copyright


    @property
    def variant_id(self) -> str:
        return self._bitstream["key"]

    @property
    def variant_cli(self) -> str:
        return self._generation["variant"]
    

    #######################################

    @property
    def generation(self) -> dict:
        return self._generation

    @generation.setter
    def generation(self, b:dict):
        if b == None:
            self._generation = None
        keys = [ "config-file", "encoder", "key", "log-file", "sequence", "variant" ]
        self._generation = { k: b[k] for k in keys }

    #######################################

    @property
    def bitstream(self) -> dict:
        return self._bitstream

    @bitstream.setter
    def bitstream(self, b:dict):
        if b == None:
            self._bitstream = None
        keys = [ "URI", "key", "md5", "size" ]
        self._bitstream = { k: b[k] for k in keys }

    #######################################

    @property
    def reconstruction(self) -> dict:
        return self._reconstruction

    @reconstruction.setter
    def reconstruction(self, b:dict):
        if b == None:
            self._reconstruction = None
        keys = [ "decoder", "log-file", "md5" ]
        self._reconstruction = { k: b[k] for k in keys }

    #######################################

    @property
    def metrics(self) -> dict:
        return self._metrics

    @metrics.setter
    def metrics(self, b:dict):
        if b == None:
            self._metrics = None
        keys = [ "Bitrate", "BitrateLog", "DecodeTime", "EncodeTime", "MS_SSIM", "UPSNR", "VMAF", "VPSNR", "YPSNR" ]
        self._metrics = { k: float(b[k]) for k in keys }

    #######################################

    @property
    def verification(self) -> dict:
        return self._verification

    @verification.setter
    def verification(self, b:dict):
        if b == None:
            self._verification = None
        keys = [ "Reports" ] #, "cross-verification-status" ]
        self._verification = { k: b[k] for k in keys }

    #######################################

    @property
    def contact(self) -> dict:
        return self._contact

    @contact.setter
    def contact(self, b:dict):
        if b == None:
            self._contact = None
        keys = [ "Company", "email" ]
        self._contact = { k: b[k] for k in keys }

    #######################################

    @property
    def copyright(self) -> str:
        return self._copyright

    @copyright.setter
    def copyright(self, c:str):
        self._copyright = c

    #######################################

    def locate_bitstream(self, anchor_dir:Path, md5_check=True) -> Path:
        assert self.bitstream != None, 'bitstream definition not found'
        assert anchor_dir.is_dir(), 'invalid anchor directory'
        fp = anchor_dir / Path(self._bitstream['URI'])
        assert fp.exists(), f'bitstream file does not exist: {fp}'
        if md5_check:
            assert self._bitstream['md5'] == md5_checksum(fp), f'md5 missmatch: {fp}'
        return fp

    def has_metric_set(self, *metrics) -> bool:
        assert (self._reconstruction != None) and (self._reconstruction['md5'] != None), 'invalid reconstruction'
        if type(self._metrics) != dict:
            return False
        for m in metrics:
            if not m in self._metrics:
                return False
        return True

    def dumps(self):
        data = {
            "Bitstream": self._bitstream,
            "Generation": self._generation,
            "Reconstruction": self._reconstruction,
            "Metrics": self._metrics,
            "Verification": self._verification,
            "copyRight": self._copyright,
            "Contact": self._contact
        }
        return json.dumps(data, indent=4)

    def save_as(self, fp:Path):
        data = self.dumps()
        with open(fp, 'w') as fo:
            fo.write(data)


#########################################################################################################


class AnchorTuple:
    
    def __init__(self, anchor_dir:Path, reference:VideoSequence, encoder_id:str, encoder_cfg:str, variants:str, anchor_key:str, description:str=None, start_frame:int=0, frame_count:int=None, dry_run:bool=False, raise_if_not_exists=True):
        assert (not raise_if_not_exists) or anchor_dir and anchor_dir.is_dir(), f'[{anchor_key}] - invalid working directory : {anchor_dir}'
        self._working_dir = anchor_dir
        self._encoder_id = encoder_id
        self._encoder_cfg = encoder_cfg
        self._reference = reference
        self._description = description
        self._anchor_key = anchor_key
        self._start_frame = start_frame
        self._frame_count = frame_count if frame_count != None else reference.frame_count

        assertion = "expecting 'variants' to be a list of integer QP values, parsed from a json string"
        try:
            data = json.loads(variants)
        except BaseException:
            raise ValueError(assertion)
        assert type(data) == list, assertion
        for v in data:
            assert type(v) == int, assertion
        self._variants = data
        self.dry_run = dry_run

    @property
    def working_dir(self) -> Path:
        return self._working_dir

    @property
    def encoder_id(self):
        return self._encoder_id

    @property
    def encoder_cfg(self) -> Path: 
        p = Path(self._encoder_cfg)
        if p.is_absolute():
            return p
        else:
            return self.working_dir.parent / 'CFG' / p

    @property
    def reference(self) -> VideoSequence:
        return self._reference

    @property
    def start_frame(self) -> int:
        return self._start_frame

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration(self) -> float:
        return self._frame_count / self.reference.frame_rate

    @property
    def description(self) -> str:
        return self._description

    @property
    def basename(self):
        return self.encoder_cfg.stem

    def iter_variants_args(self) -> Generator[Tuple[str, list], None, None]:
        """
        yields (variant_id, variant_encoder_args)
        """
        for qp in self._variants:
            yield self._anchor_key.replace('<QP>', str(qp)), (qp,)
    
    @property
    def anchor_key(self):
        return self._anchor_key



#########################################################################################################

def __fix_sidecar_meta(loc):
    # @FIXME: this file doesn't follow the pattern
    if loc != 'Baolei-Man':
        meta = f'{loc}/{loc}.json'
    else:
        meta = f'{loc}/{str(loc).lower()}.json'
    return meta

def iter_ref_locations(reference_list:Path) -> Iterable[str]:
    refs = []
    with open(reference_list, 'r', encoding=ENCODING) as fo:
        for row in csv.DictReader(fo):
            refs.append( __fix_sidecar_meta(row[RefSequenceList.LOC]) )
    return refs

def reference_sequences_dict(reference_list:Path, root_dir:Path=Path('.')) -> Dict[str, VideoSequence]:
    refs = {}
    with open(reference_list, 'r', encoding=ENCODING) as fo:
        for row in csv.DictReader(fo):
            meta = root_dir / __fix_sidecar_meta(row[RefSequenceList.LOC])
            vs = VideoSequence.from_sidecar_metadata(meta)
            assert (vs.frame_count / vs.frame_rate) == float(row[RefSequenceList.DUR]), f'(frame_count / frame_rate) != expected duration'
            refs[row[RefSequenceList.KEY]] = vs
    return refs

def iter_anchors_csv(anchor_list:Path):
    with open(anchor_list, 'r', encoding=ENCODING) as fo:
        for row in csv.DictReader(fo):
            yield row

def iter_anchors(anchor_list:Path, refs:Dict[str, VideoSequence], scenario_dir:Path, cfg_dir=None, keys:Iterable[str]=None, raise_if_not_exists=True) -> Iterable[AnchorTuple]:
    anchors = []
    with open(anchor_list, 'r', encoding=ENCODING) as fo:
        for row in csv.DictReader(fo):
            anchor_key = row[AnchorList.KEY]
            if (keys != None) and (anchor_key not in keys):
                continue
            seq = refs[row[AnchorList.REF_SEQ]]
            description = row[AnchorList.CLAUSE]
            encoder_id = row[AnchorList.REF_ENC] if AnchorList.REF_ENC in row else row[AnchorList.TEST_ENC] # eg. HM16.22, 
            encoder_cfg = str(row[AnchorList.CFG]).lower() + '.cfg' # eg. S3-HM-01, no directory context specified
            if cfg_dir:
                encoder_cfg = Path(cfg_dir / encoder_cfg).resolve()
            variants = row[AnchorList.VARIANTS]
            anchor_dir = scenario_dir / row[AnchorList.KEY]
            bitsream_key_template = row[AnchorList.VARIANT_KEY]
            anchors.append(
                AnchorTuple(anchor_dir, seq, encoder_id, encoder_cfg, variants, bitsream_key_template, description, seq.start_frame, seq.frame_count, raise_if_not_exists=raise_if_not_exists)
            )
    return anchors

def iter_variants(a:AnchorTuple) -> Iterable[Tuple[Path, VariantData]]:
    """ 
    yields (a.working_dir/variant.json, VariantData)
    """
    for variant_id, _ in a.iter_variants_args():
        vfp = a.working_dir / f'{variant_id}.json'
        data = None
        if vfp.exists():
            data = VariantData.load(vfp)
        yield vfp, data
        

def check_anchor_data(a):
    try:
        assert a.working_dir.exists(), f'working dir not found: {a.working_dir}'
        assert a.reference.path.exists(), f'reference sequence not found: {a.reference.path}'
        assert a.encoder_cfg.exists(), f'encoder config not found: {a.encoder_cfg}'
        assert parse_encoding_bitdepth(a.encoder_cfg) == 10, f'encoder config error - InternalBitDepth'
        if a.reference.bit_depth != 10:
            assert a.reference.bit_depth == 8, f'unexpected bit depth: {a.reference.bit_depth}'
        conv = conversion_path(a.reference.path, '10bit')
        assert conv.exists(), '10bit conversion of reference sequence is missing'
        return None
    except AssertionError as e:
        return e


def check_variant_data(v, a, md5_check=False):
    try:
        assert 'verified' in v.verification, 'variant data not verified'
        v.locate_bitstream(a.working_dir, md5_check=md5_check)
        return None
    except BaseException as e:
        return e
