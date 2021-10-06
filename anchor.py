import json, csv, math, enum
from pathlib import Path
from typing import Dict, Generator, Tuple, Iterable
from itertools import chain

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


class M:

    def __init__(self, *args) -> None:
        assert len(args) and (len(args) < 2)
        self.key = str(args[0]).lower()
        self.json_key = str(args[0])
        self.csv_key = str(args[-1])

    def __repr__(self):
        return self.key

    def __str__(self):
        return self.key

class Metric(enum.Enum):

    @property
    def csv_key(self):
        return self.value.csv_key

    @property
    def json_key(self):
        return self.value.json_key

    @property
    def key(self):
        return self.value.key

    PSNR        = M( "PSNR" )
    PSNR_Y      = M( "YPSNR" )
    PSNR_U      = M( "UPSNR" )
    PSNR_V      = M( "VPSNR" )
    MSSSIM      = M( "MS_SSIM" )
    VMAF        = M( "VMAF" )
    BITRATE     = M( "Bitrate" )
    BITRATELOG  = M( "BitrateLog" )
    ENCODETIME  = M( "EncodeTime" )
    DECODETIME  = M( "DecodeTime" )

    @classmethod
    def json_dict(cls, v:'VariantMetricSet'):
        j = { m.key: m.json_key for m in cls }
        return { j[k]: v for (k, v) in v.items() }

    @classmethod
    def csv_dict(cls, v:'VariantMetricSet'):
        c = { m.key: m.csv_key for m in cls }
        return { c[k]: v for (k, v) in v.items() }


class ReconstructionMeta:
    def __init__(self, decoder_id:str, reconstructed:Path, decoder_log:Path, md5=True):
        self.decoder_id = decoder_id
        self.reconstructed = reconstructed
        if md5:
            print(f'computing md5 for {reconstructed}')
            self.reconstructed_md5 = md5_checksum(self.reconstructed)
        else:
            self.reconstructed_md5 = None
        self.decoder_log = decoder_log

    def to_dict(self) -> dict:
        return {
            "decoder": self.decoder_id,
            "log-file": self.decoder_log.name if hasattr(self.decoder_log, 'decoder_log') else None,
            "md5": self.reconstructed_md5
        }


class VariantMetricSet(dict):

    def compute_avg_psnr(self, strict=False):
        try:
            [Y, U, V] = [self.get(k.key) for k in [Metric.PSNR_Y, Metric.PSNR_U, Metric.PSNR_V]]
            psnr = ((6*Y)+U+V)/8
            self.update({ Metric.PSNR.key: psnr })
            return psnr
        except BaseException as e:
            if strict:
                raise
            self.update({ Metric.PSNR.key: str(e) })
            return None

    __slots__ = ()

    @staticmethod
    def _process_args(mapping=(), **kwargs):
        if hasattr(mapping, 'items'):
            mapping = mapping.items()
        return [(to_lower(k), v) for k, v in chain(mapping, kwargs.items())]

    def __init__(self, mapping=(), **kwargs):
        args = self._process_args(mapping, **kwargs)
        super(VariantMetricSet, self).__init__(mapping=args)

    def __getitem__(self, k):
        return super(VariantMetricSet, self).__getitem__(to_lower(k))

    def __setitem__(self, k, v):
        return super(VariantMetricSet, self).__setitem__(to_lower(k), v)

    def __delitem__(self, k):
        return super(VariantMetricSet, self).__delitem__(to_lower(k))

    def get(self, k, default=None):
        return super(VariantMetricSet, self).get(to_lower(k), default)

    def setdefault(self, k, default=None):
        return super(VariantMetricSet, self).setdefault(to_lower(k), default)

    def pop(self, k, v):
        return super(VariantMetricSet, self).pop(to_lower(k), v)

    def update(self, mapping=(), **kwargs):
        args = self._process_args(mapping, **kwargs)
        super(VariantMetricSet, self).update(mapping=args)

    def __contains__(self, k):
        return super(VariantMetricSet, self).__contains__(to_lower(k))

    def copy(self):
        return type(self)(self)

    @classmethod
    def fromkeys(cls, keys, v=None):
        return super(VariantMetricSet, cls).fromkeys((to_lower(k) for k in keys), v)

    def __repr__(self):
        return '{0}({1})'.format(type(self).__name__, super(VariantMetricSet, self).__repr__())


def to_lower(maybe_str):
    return maybe_str.lower() if isinstance(maybe_str, str) else maybe_str


class VariantData:

    @classmethod
    def load(cls, fp:Path) -> 'VariantData':
        assert fp.exists(), f'File not found: {fp}'
        with open(fp, 'r') as fo:
            data = json.load(fo)
            generation = data.get("Generation", None)
            bitstream = data.get("Bitstream", None)
            reconstruction = data.get("Reconstruction", None)

            d = data.get("Metrics", None)
            metrics = None
            if d != None:
                metrics = VariantMetricSet()
                for k, v in d.items():
                    try:
                        metrics[k] = float(v)
                    except BaseException:
                        metrics[k] = None
                if Metric.PSNR.key not in metrics:
                    metrics.compute_avg_psnr(strict=False)
            
            verification = data.get("Verification", None)
            contact = data.get("contact", None)
            copyright = data.get("copyRight", None)
            return VariantData(generation, bitstream, reconstruction, metrics, verification, contact, copyright)

    @classmethod
    def new(cls, a:'AnchorTuple', variant_id:str, variant_cli:str, encoder_log:Path, bitstream_fp:Path, reconstruction:'ReconstructionMeta') -> 'VariantData':
        generation = {
            "key": variant_id,
            "sequence": a.reference.path.name,
            "encoder": a.encoder_id,
            "config-file": a.encoder_cfg.name,
            "variant": variant_cli,
            "log-file": encoder_log.name
        }
        bitstream = {
            "key": variant_id,
            "URI": bitstream_fp.name,
            "md5": md5_checksum(bitstream_fp),
            "size": bitstream_fp.stat().st_size
        }
        contact = {
            'Company': a.reference.contact['Company'],
            'e-mail': a.reference.contact['e-mail']
        }
        copyright = a.reference.copyright
        return VariantData(generation, bitstream, reconstruction.to_dict(), None, None, contact, copyright)

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
    def metrics(self) -> VariantMetricSet:
        return self._metrics

    @metrics.setter
    def metrics(self, b:VariantMetricSet):
        self._metrics = b


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


#########################################################################################################


class AnchorTuple:
    
    def __init__(self, anchor_dir:Path, reference:VideoSequence, encoder_id:str, encoder_cfg:str, variants:str, anchor_key:str, description:str=None, start_frame:int=0, frame_count:int=None, dry_run:bool=False):
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

    @working_dir.setter
    def working_dir(self, wd:Path):
        if not issubclass(type(wd), Path):
            raise ValueError('Expecting a valid pathlib.Path')
        self._working_dir = wd

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
        """generator to iterate over (variant_id, variant_encoder_args)
            variant_id can be used to locate variant bitstream json file.
            variant_encoder_args is parsed by encoders implementations, 
                eg. using by python's shlex module.
                variant_encoder_args is stored as is in the bitstream json metadata.
        """
        for qp in self._variants:
            yield self._anchor_key.replace('<QP>', str(qp)), f'-qp {qp}'

    @property
    def anchor_key(self):
        return self._anchor_key

    def relative_path(self, p:Path) -> Path:
        if p.is_relative_to(self.working_dir):
            return p.relative_to(self.working_dir)
        else:
            return p

    def locate_bitstream(self, vd:VariantData, md5_check=True) -> Path:
        assert vd.bitstream != None, 'bitstream definition not found'
        assert self.working_dir.is_dir(), 'invalid anchor directory'
        fp = self.working_dir / Path(vd.bitstream['URI'])
        assert fp.exists(), f'bitstream file does not exist: {fp}'
        if md5_check:
            assert vd.bitstream['md5'] == md5_checksum(fp), f'md5 missmatch: {fp}'
        return fp


#########################################################################################################

def ref_location(row) -> str:
    loc = row[RefSequenceList.LOC]
    return f'{loc}/{loc}.json'

def iter_ref_locations(reference_list:Path) -> Iterable[str]:
    refs = []
    with open(reference_list, 'r', encoding=ENCODING) as fo:
        for row in csv.DictReader(fo):
            refs.append( ref_location(row) )
    return refs

def reference_sequences_dict(reference_list:Path, root_dir:Path=Path('.'), raises=False) -> Dict[str, VideoSequence]:
    """produces a dict mapping a reference sequence key 
        to a VideoSequence if the sequence exists - None if it doesn't exist, or raises is True. 
    """
    refs = {}
    with open(reference_list, 'r', encoding=ENCODING) as fo:
        for row in csv.DictReader(fo):
            meta = root_dir / ref_location(row)
            k = row[RefSequenceList.KEY]
            if not meta.exists():
                if raises:
                    raise FileNotFoundError(str(meta.resolve()))
                refs[k] = None
                continue
            vs = VideoSequence.from_sidecar_metadata(meta)
            if vs.sequence:
                vs.sequence['Key'] = k
            dur = float(row[RefSequenceList.DUR])
            if not math.isclose( vs.frame_count/vs.frame_rate, dur):
                print(f'(frame_count:{vs.frame_count} / frame_rate:{vs.frame_rate}):{(vs.frame_count / vs.frame_rate)} != "duration:{dur} found in `scenario/reference-sequence.csv` file for `{k}`"')
            refs[k] = vs
    return refs

def iter_anchors_csv(anchor_list:Path) -> Dict[str, str]:
    with open(anchor_list, 'r', encoding=ENCODING) as fo:
        for row in csv.DictReader(fo):
            yield row

def iter_anchors(anchor_list:Path, refs:Dict[str, VideoSequence], scenario_dir:Path, cfg_dir=None, keys:Iterable[str]=None, cfg_keys:Iterable[str]=None) -> Iterable[AnchorTuple]:
    anchors = []
    with open(anchor_list, 'r', encoding=ENCODING) as fo:
        for row in csv.DictReader(fo):
            anchor_key = row[AnchorList.KEY]
            if (keys != None) and (anchor_key not in keys):
                continue
            seq = refs[row[AnchorList.REF_SEQ]]
            description = row[AnchorList.CLAUSE]
            encoder_id = row[AnchorList.REF_ENC] if AnchorList.REF_ENC in row else row[AnchorList.TEST_ENC] # eg. HM16.22, 
            encoder_cfg_key = str(row[AnchorList.CFG])
            encoder_cfg = encoder_cfg_key.lower() + '.cfg' # eg. S3-HM-01, no directory context specified
            if (cfg_keys != None) and (encoder_cfg_key not in cfg_keys):
                continue
            if cfg_dir:
                encoder_cfg = Path(cfg_dir / encoder_cfg).resolve()
            variants = row[AnchorList.VARIANTS]
            anchor_dir = scenario_dir / row[AnchorList.KEY]
            bitsream_key_template = row[AnchorList.VARIANT_KEY]
            anchors.append(
                AnchorTuple(anchor_dir, seq, encoder_id, encoder_cfg, variants, bitsream_key_template, description, seq.start_frame, seq.frame_count)
            )
    return anchors

def iter_variants(a:AnchorTuple) -> Iterable[Tuple[Path, VariantData]]:
    """Iterate the variant bitstream's json metadata path of the anchor, 
        loads and returns VariantData if the metadata file exists.
    """
    for variant_id, _ in a.iter_variants_args():
        vfp = a.working_dir / f'{variant_id}.json'
        data = None
        if vfp.exists():
            data = VariantData.load(vfp)
        yield vfp, data
