import json
import csv
import math
import enum
import argparse
import logging

from pathlib import Path
from typing import Dict, Generator, Tuple, Iterable, List
from itertools import chain

from utils import VideoSequence, md5_checksum

ENCODING = 'utf-8-sig'


def anchor_path_from_key(key: str, root_dir:Path ) -> Path:
    arr = key.split('-')
    if len(arr) == 3:
        scenario, _, enc = arr
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
            raise ValueError(f'Invalid scenario key "{scenario}" in {key}')
        return root_dir / f'Bitstreams/{scenario}/{enc}/{key}/'
    
    raise ValueError(f'Invalid anchor key "{key}"')


def encoder_config_path(key: str, root_dir:Path ) -> Path:
    arr = key.split('-')
    if len(arr) == 3:
        scenario, enc, cfg = arr
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
            raise ValueError(f'Invalid scenario key "{scenario}" in {key}')
        
        if enc == 'HM':
            enc = '265'
        elif enc == 'SCM':
            enc = '265'
        elif enc == 'JM':
            enc = '264'

        return root_dir / f'Bitstreams/{scenario}/{enc}/CFG/{key}.cfg'
    
    raise ValueError(f'Invalid anchor key "{key}"')


# scenario/anchors.csv
class AnchorList:
    KEY = '#Key'  # directory where anchor is stored
    CLAUSE = 'Clause'
    REF_SEQ = 'Reference Sequence'
    REF_ENC = 'Reference_Encoder'
    TEST_ENC = 'Test_Encoder'
    CFG = 'Configuration'
    VARIANTS = 'Variations'
    VARIANT_KEY = 'Anchor_Key'  # template for variant json filename


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

    PSNR = M("PSNR")
    PSNR_Y = M("YPSNR")
    PSNR_U = M("UPSNR")
    PSNR_V = M("VPSNR")
    MSSSIM = M("MS_SSIM")
    VMAF = M("VMAF")
    BITRATE = M("Bitrate")
    BITRATELOG = M("BitrateLog")
    ENCODETIME = M("EncodeTime")
    DECODETIME = M("DecodeTime")
    DELTAE100 = M("DELTAE100")
    WTPSNR = M("WTPSNR")
    WTPSNR_Y = M("WTPSNR_Y")
    WTPSNR_U = M("WTPSNR_U")
    WTPSNR_V = M("WTPSNR_V")
    GSSIM = M("GSSIM")
    PSNR_DE0100 = M("PSNR_DE0100")
    PSNR_MD0100 = M("PSNR_MD0100")
    PSNR_L0100 = M("PSNR_L0100")

    @classmethod
    def json_dict(cls, v: 'VariantMetricSet'):
        j = {m.key: m.json_key for m in cls}
        return {j[k]: v for (k, v) in v.items()}

    @classmethod
    def csv_dict(cls, v: 'VariantMetricSet'):
        c = {m.key: m.csv_key for m in cls}
        return {c[k]: v for (k, v) in v.items()}


class ReconstructionMeta:
    def __init__(self, decoder_id: str, reconstructed: Path, decoder_log: Path, md5=True):
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
            psnr = ((6 * Y) + U + V) / 8
            u = {Metric.PSNR.key: psnr}
            self.update(u)
            return psnr
        except BaseException as e:
            if strict:
                raise
            u = {Metric.PSNR.key: str(e)}
            self.update(u)
            return None

    __slots__ = ()

    @classmethod
    def _process_args(cls, mapping=(), **kwargs):
        if hasattr(mapping, 'items'):
            mapping = mapping.items()
        d = {}
        for K, v in chain(mapping, kwargs.items()):
            k = to_lower(K)
            d[k] = v
        return d

    def __init__(self, mapping=(), **kwargs):
        args = self._process_args(mapping, **kwargs)
        super(VariantMetricSet, self).__init__(args)

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
        super(VariantMetricSet, self).update(args)

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
    def load(cls, fp: Path) -> 'VariantData':
        if not fp.exists():
            raise FileNotFoundError(f'File not found: {fp}')

        with open(fp, 'r') as fo:
            data = json.load(fo)
            generation = data.get("Generation", None)
            bitstream = data.get("Bitstream", None)
            reconstruction = data.get("Reconstruction", None)

            d = data.get("Metrics", None)
            metrics = None
            if d is not None:
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
    def new(cls, a: 'AnchorTuple', variant_id: str, variant_cli: str, encoder_log: Path, bitstream_fp: Path, reconstruction: 'ReconstructionMeta') -> 'VariantData':
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

    def save_as(self, fp: Path):
        data = self.dumps()
        with open(fp, 'w') as fo:
            fo.write(data)

    def __init__(self, generation: dict = None, bitstream: dict = None, reconstruction: dict = None, metrics: dict = None, verification: dict = None, contact: dict = None, copyright: str = ''):
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

    @property
    def generation(self) -> dict:
        return self._generation

    @generation.setter
    def generation(self, b: dict):
        if b is None:
            self._generation = None
        keys = ["config-file", "encoder", "key", "log-file", "sequence", "variant"]
        self._generation = {k: b[k] for k in keys}

    @property
    def bitstream(self) -> dict:
        return self._bitstream

    @bitstream.setter
    def bitstream(self, b: dict):
        if b is None:
            self._bitstream = None
        keys = ["URI", "key", "md5", "size"]
        self._bitstream = {k: b[k] for k in keys}

    @property
    def reconstruction(self) -> dict:
        return self._reconstruction

    @reconstruction.setter
    def reconstruction(self, b: dict):
        if b is None:
            self._reconstruction = None
        keys = ["decoder", "log-file", "md5"]
        self._reconstruction = {k: b[k] for k in keys}

    @property
    def metrics(self) -> VariantMetricSet:
        return self._metrics

    @metrics.setter
    def metrics(self, b: VariantMetricSet):
        self._metrics = b

    @property
    def verification(self) -> dict:
        return self._verification

    @verification.setter
    def verification(self, b: dict):
        if b is None:
            self._verification = None
        keys = ["Reports"]  # , "cross-verification-status"]
        self._verification = {k: b[k] for k in keys}

    @property
    def contact(self) -> dict:
        return self._contact

    @contact.setter
    def contact(self, b: dict):
        if b is None:
            self._contact = None
        keys = ["Company", "email"]
        self._contact = {k: b[k] for k in keys}

    @property
    def copyright(self) -> str:
        return self._copyright

    @copyright.setter
    def copyright(self, c: str):
        self._copyright = c

    def locate_bitstream(self, anchor_dir: Path, md5_check=True) -> Path:
        assert self.bitstream is not None, 'bitstream definition not found'
        assert anchor_dir.is_dir(), 'invalid anchor directory'
        fp = anchor_dir / Path(self._bitstream['URI'])
        assert fp.exists(), f'bitstream file does not exist: {fp}'
        if md5_check:
            assert self._bitstream['md5'] == md5_checksum(fp), f'md5 missmatch: {fp}'
        return fp

    def has_metric_set(self, *metrics) -> bool:
        assert (self._reconstruction is not None) and (self._reconstruction['md5'] is not None), 'invalid reconstruction'
        if type(self._metrics) != dict:
            return False
        for m in metrics:
            if m not in self._metrics:
                return False
        return True


class AnchorTuple:

    @classmethod
    def load(cls, anchor_key:str, root_dir=Path('/data')) -> 'AnchorTuple':
        anchor_dir = anchor_path_from_key(anchor_key, root_dir)
        ctx = AnchorTupleCtx(scenario_dir=anchor_dir.parent)
        anchors = ctx.iter_anchors(keys=[anchor_key])
        assert len(anchors) == 1, 'duplicate anchor key found in streams.csv'
        return anchors[0]

    @classmethod
    def iter_cfg_anchors(cls, config_key, root_dir=Path('/data')):
        test_dir = encoder_config_path(config_key, root_dir).parent.parent
        ctx = AnchorTupleCtx(scenario_dir=test_dir)
        anchors = ctx.iter_anchors(cfg_keys=[config_key])
        assert len(anchors), f'No anchor/test found for encoder config {config_key}'
        return anchors


    def __init__(self, anchor_dir: Path, reference: VideoSequence, encoder_id: str, encoder_cfg: str, variants: str, anchor_key: str, description: str = None, start_frame: int = 0, frame_count: int = None, dry_run: bool = False):
        self._working_dir = anchor_dir
        self._encoder_id = encoder_id
        self._encoder_cfg = encoder_cfg
        self._reference = reference
        self._description = description
        self._anchor_key = anchor_key
        self._start_frame = start_frame
        self._frame_count = frame_count if frame_count is not None else reference.frame_count

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
    def working_dir(self, wd: Path):
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

    def relative_path(self, p: Path) -> Path:
        if p.is_relative_to(self.working_dir):
            return p.relative_to(self.working_dir)
        else:
            return p

    def locate_bitstream(self, vd: VariantData, md5_check=True) -> Path:
        assert vd.bitstream is not None, 'bitstream definition not found'
        assert self.working_dir.is_dir(), 'invalid anchor directory'
        fp = self.working_dir / Path(vd.bitstream['URI'])
        assert fp.exists(), f'bitstream file does not exist: {fp}'
        if md5_check:
            assert vd.bitstream['md5'] == md5_checksum(fp), f'md5 missmatch: {fp}'
        return fp


def ref_location(row) -> str:
    loc = row[RefSequenceList.LOC]
    name = row[RefSequenceList.NAME]
    return f'{loc}/{name}.json'


def iter_ref_locations(reference_list: Path) -> Iterable[str]:
    refs = []
    with open(reference_list, 'r', encoding=ENCODING) as fo:
        for row in csv.DictReader(fo):
            refs.append(ref_location(row))
    return refs


def reference_sequences_dict(reference_list: Path, root_dir: Path = Path('.'), raises=False) -> Dict[str, VideoSequence]:
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
            if not math.isclose(vs.frame_count / vs.frame_rate, dur):
                print(f'# (frame_count:{vs.frame_count} / frame_rate:{vs.frame_rate}):{(vs.frame_count / vs.frame_rate)} != "duration:{dur} found in `scenario/reference-sequence.csv` file for `{k}`"')
            refs[k] = vs
    return refs


def iter_anchors_csv(anchor_list: Path) -> Dict[str, str]:
    with open(anchor_list, 'r', encoding=ENCODING) as fo:
        for row in csv.DictReader(fo):
            yield row


def iter_anchors(anchor_list: Path, refs: Dict[str, VideoSequence], scenario_dir: Path, cfg_dir=None, keys: Iterable[str] = None, cfg_keys: Iterable[str] = None) -> Iterable[AnchorTuple]:
    anchors = []
    with open(anchor_list, 'r', encoding=ENCODING) as fo:
        for row in csv.DictReader(fo):
            anchor_key = row[AnchorList.KEY]
            if (keys is not None) and (anchor_key not in keys):
                continue
            seq = refs[row[AnchorList.REF_SEQ]]
            assert seq is not None, f'[{anchor_key}] reference sequence not found: {row[AnchorList.REF_SEQ]}'
            description = row[AnchorList.CLAUSE]
            encoder_id = row[AnchorList.REF_ENC] if AnchorList.REF_ENC in row else row[AnchorList.TEST_ENC]  # eg. HM16.22
            encoder_cfg_key = str(row[AnchorList.CFG])
            encoder_cfg = encoder_cfg_key.lower() + '.cfg'  # eg. S3-HM-01, no directory context specified
            if (cfg_keys is not None) and (encoder_cfg_key not in cfg_keys):
                continue
            if cfg_dir:
                encoder_cfg = Path(cfg_dir / encoder_cfg).resolve()
            variants = row[AnchorList.VARIANTS]
            anchor_dir = scenario_dir / row[AnchorList.KEY]
            bitsream_key_template = row[AnchorList.VARIANT_KEY]
            anchors.append(
                AnchorTuple(anchor_dir, seq, encoder_id, encoder_cfg, variants, bitsream_key_template, description, seq.start_frame, seq.frame_count))
    return anchors


def iter_variants(a: AnchorTuple) -> Iterable[Tuple[Path, VariantData]]:
    """Iterate the variant bitstream's json metadata path of the anchor,
        loads and returns VariantData if the metadata file exists.
    """
    for variant_id, _ in a.iter_variants_args():
        vfp = a.working_dir / f'{variant_id}.json'
        data = None
        if vfp.exists():
            data = VariantData.load(vfp)
        yield vfp, data



class AnchorTupleCtx:

    @classmethod
    def parse_args(cls, parser: argparse.ArgumentParser = None, scenario_dir=None,) -> 'AnchorTupleCtx':
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument('--scenario_dir', required=True, default=None, type=str, help='scenario directory, eg. Scenario-3-Screen/265')
        parser.add_argument('-k', '--key', required=False, type=str, default=None, help='an optional anchor key')
        parser.add_argument('-a', '--anchors-list', required=False, type=str, default='./streams.csv', help='streams.csv file containing the list of anchors for a scenario')
        parser.add_argument('-s', '--sequences-list', required=False, type=str, default='../reference-sequence.csv', help='sequences.csv file containing the list of reference raw sequences')
        parser.add_argument('--sequences_dir', required=False, type=str, help='the directory containing the reference sequences')
        parser.add_argument('--dry-run', action='store_true', default=False)
        parser.add_argument('--dl-streams', action='store_true', default=False, help='download streams')
        parser.add_argument('--dl-ref-sequences', action='store_true', default=False, help='download reference sequences')
        cli_args = parser.parse_args()
        return cls(cli_args=cli_args)

    @classmethod
    def from_anchor_directory(cls, ad: Path) -> Tuple['AnchorTupleCtx', str]:
        assert ad.is_dir(), f'invalid directory: {ad}'
        ad = ad.resolve()
        return cls(scenario_dir=ad.parent), ad.name

    @classmethod
    def from_encoder_directory(cls, ad: Path) -> Tuple['AnchorTupleCtx', str]:
        assert ad.is_dir(), f'invalid directory: {ad}'
        ad = ad.resolve()
        return cls(scenario_dir=ad.parent), ad.name

    def __init__(self, cli_args=None, **kwargs):

        if cli_args:
            scenario_dir = Path(cli_args.scenario_dir)
            if not scenario_dir.exists():
                scenario_dir.mkdir(parents=True)

            references_csv = scenario_dir / Path(cli_args.sequences_list)
            if not references_csv.exists():
                logging.info(f'reference-sequence.csv not found {references_csv}')

            anchors_csv = scenario_dir / cli_args.anchors_list
            if not anchors_csv.exists():
                logging.info(f'streams.csv list not found {anchors_csv}')

            sequences_dir = scenario_dir / '../../../ReferenceSequences'
            if cli_args.sequences_dir is not None:
                sequences_dir = Path(cli_args.sequences_dir).resolve()
                assert sequences_dir.is_dir(), f'invalid sequence directory {sequences_dir}'

            self.scenario_dir = scenario_dir
            self.anchors_csv = anchors_csv
            self.references_csv = references_csv
            self.sequences_dir = sequences_dir
            self.dry_run = cli_args.dry_run

            self.dl_streams = cli_args.dl_streams
            self.dl_ref_sequences = not self.dl_streams
            self.key = cli_args.key

        else:
            self.scenario_dir = Path(kwargs['scenario_dir'])
            if 'anchors_csv' in kwargs:
                self.anchors_csv = kwargs['anchors_csv']
            else:
                self.anchors_csv = self.scenario_dir / 'streams.csv'

            if 'references_csv' in kwargs:
                self.references_csv = kwargs['references_csv']
            else:
                self.references_csv = self.scenario_dir.parent / 'reference-sequence.csv'

            if 'sequences_dir' in kwargs:
                self.sequences_dir = kwargs['sequences_dir']
            else:
                self.sequences_dir = self.base_dir() / 'ReferenceSequences'

            if 'metrics_dir' in kwargs:
                self.metrics_dir = kwargs['metrics_dir']
            else:
                self.metrics_dir = self.scenario_dir / 'Metrics'

            if 'dry_run' in kwargs:
                self.dry_run = bool(kwargs['dry_run'])
            else:
                self.dry_run = False

    def base_dir(self) -> Path:
        # scenario_dir is $base_dir/Bitstreams/$scenario/$codec/
        return self.scenario_dir.parent.parent.parent

    def iter_ref_sequences(self) -> Generator[Tuple[str, VideoSequence], None, None]:
        for key, video_sequence in reference_sequences_dict(self.references_csv, self.sequences_dir).items():
            yield key, video_sequence

    def iter_anchors(self, keys: List[str] = None, cfg_keys: List[str] = None) -> Iterable[AnchorTuple]:
        if (keys is None) and hasattr(self, "key"):
            keys = [self.key]
        refs = reference_sequences_dict(self.references_csv, self.sequences_dir)
        return iter_anchors(self.anchors_csv, refs, self.scenario_dir, keys=keys, cfg_keys=cfg_keys)
