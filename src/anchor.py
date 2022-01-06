import json
import csv
import logging
from pathlib import Path
from typing import Dict, Generator, Tuple, Iterable, List
from itertools import chain
from utils import md5_checksum
from constants import encoder_cfg_path, Metric, RefSequenceList, AnchorList, HDR_METRICS, SDR_METRICS, ENCODING
from sequences import VideoSequence, TransferFunction



class ReconstructionMeta:
    def __init__(self, decoder_id: str, reconstructed: Path, decoder_log: Path, md5=True):
        self.decoder_id = decoder_id
        self.reconstructed = reconstructed
        if md5:
            logging.info(f'computing md5 for {reconstructed}')
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

    def compute_avg_psnr(self, raises=False):
        try:
            [Y, U, V] = [self.get(k) for k in [Metric.PSNR_Y, Metric.PSNR_U, Metric.PSNR_V]]
            psnr = ((6 * Y) + U + V) / 8
            u = {Metric.PSNR: psnr}
            self.update(u)
            return psnr
        except BaseException as e:
            if raises:
                raise
            u = {Metric.PSNR: 0}
            self.update(u)
            return None

    def compute_avg_wpsnr(self, raises=False):
        try:
            [Y, U, V] = [self.get(k) for k in [Metric.WTPSNR_Y, Metric.WTPSNR_U, Metric.WTPSNR_V]]
            psnr = ((6 * Y) + U + V) / 8
            u = {Metric.WTPSNR: psnr}
            self.update(u)
            return psnr
        except BaseException as e:
            if raises:
                raise
            u = {Metric.WTPSNR: 0}
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
    def load(cls, fp: Path, variant_id:str) -> 'VariantData':
        if not fp.exists():
            raise FileNotFoundError(f'File not found: {fp}')

        with open(fp, 'r') as fo:
            data = json.load(fo)
            qp = variant_id.split('-')[-1]
            generation = data.get("Generation", {})
            if ("variant" in generation) and (generation["variant"] != qp):
                err = generation["variant"]
                logging.info(f"Invalid generation[variant] {err} instead of {qp} found in {fp}")
            generation["variant"] = qp

            bitstream = data.get("Bitstream", {})
            if ("key" in bitstream) and (bitstream["key"] != variant_id):
                err = bitstream["key"]
                logging.info(f"Invalid bitstream[key] {err} instead of {variant_id} found in {fp}")
            bitstream["key"] = variant_id

            d = data.get("Metrics", None)
            metrics = None
            if (d is not None) and (type(d) is dict) :
                metrics = VariantMetricSet()
                for k, v in d.items():
                    try:
                        m = Metric.from_json_key(k)
                        assert m, f'Unknown metric key "{k}" used in json metadata: {k}'
                        metrics[m] = float(v)
                    except BaseException as e:
                        logging.error(f'{fp.name} - {e}')

                if (Metric.PSNR_Y in metrics) and (Metric.PSNR not in metrics):
                    metrics.compute_avg_psnr()

                if (Metric.WTPSNR_Y in metrics) and (Metric.WTPSNR not in metrics):
                    metrics.compute_avg_wpsnr()
            
            verification = data.get("Verification", None)
            contact = data.get("contact", None)
            copyright = data.get("copyRight", None)
            
            reconstruction = data.get("Reconstruction", None)
            """
            bitstream['URI'] = f'{variant_id}.bin'
            generation['log-file'] = f'{variant_id}.encoder.log'

            vd = VariantData(generation, bitstream, reconstruction, metrics, verification, contact, copyright)
            del vd.generation['key']
            vd.save_as(fp)
            return vd
            """
            return VariantData(generation, bitstream, reconstruction, metrics, verification, contact, copyright)

    @classmethod
    def new(cls, a: 'AnchorTuple', variant_id: str, variant_qp: str, encoder_log: Path, bitstream_fp: Path, reconstruction: 'ReconstructionMeta') -> 'VariantData':
        generation = {
            "sequence": a.reference.path.name,
            "encoder": a.encoder_id,
            "config-file": a.encoder_cfg.name,
            "variant": variant_qp,
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
        metrics = None if self._metrics is None else { k.json_key: v for k, v in self._metrics.items() }
        data = {
            "Bitstream": self._bitstream,
            "Generation": self._generation,
            "Reconstruction": self._reconstruction,
            "Metrics": metrics,
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
    def variant_qp(self) -> str:
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
    def locate(cls, anchor_key: str, parent_dir: Path) -> Path:
        arr = anchor_key.split('-')
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
                return parent_dir / anchor_key
            return parent_dir / f'{scenario}/{enc}/{anchor_key}/'
        else:
            return parent_dir / anchor_key

    @classmethod
    def load(cls, anchor_key:str, bitstreams_dir:Path, sequences_dir:Path) -> 'AnchorTuple':
        anchor_dir = cls.locate(anchor_key, bitstreams_dir)
        codec_dir = anchor_dir.parent
        scenario_dir = codec_dir.parent
        sequences = reference_sequences_dict(scenario_dir / 'reference-sequence.csv', sequences_dir) if sequences_dir is not None else None
        streams_csv = codec_dir / 'streams.csv'
        anchors = iter_anchors(streams_csv, sequences=sequences, keys=[anchor_key])
        if len(anchors) > 1:
            raise Exception(f'duplicate anchor key {anchor_key} found in streams.csv')
        elif len(anchors) == 0:
            raise Exception(f'{anchor_key} not found in {streams_csv}')
        return anchors[0]

    @classmethod
    def iter_cfg_anchors(cls, config_key, bitstreams_dir:Path, sequences_dir:Path):
        test_dir = encoder_cfg_path(config_key, bitstreams_dir).parent.parent
        sequences = reference_sequences_dict(test_dir.parent / 'reference-sequence.csv', sequences_dir) if sequences_dir is not None else None
        streams_csv = test_dir / 'streams.csv'
        anchors = iter_anchors(streams_csv, sequences=sequences, cfg_keys=[config_key])
        if len(anchors) == 0:
            raise Exception(f'No anchor/test found for encoder config {config_key} in {streams_csv}')
        return anchors


    def __init__(self, anchor_dir: Path, reference: VideoSequence, encoder_id: str, encoder_cfg: str, variants: str, anchor_key: str, description: str = None, dry_run: bool = False):
        self._working_dir = anchor_dir
        self._encoder_id = encoder_id
        self._encoder_cfg = encoder_cfg
        self._reference = reference
        self._description = description
        self._anchor_key = anchor_key

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
        assert self._reference is not None, 'reference sequence not loaded'
        return self._reference.start_frame

    @property
    def frame_count(self) -> int:
        assert self._reference is not None, 'reference sequence not loaded'
        return self._reference.frame_count

    @property
    def duration(self) -> float:
        return self.frame_count / self.reference.frame_rate

    @property
    def description(self) -> str:
        return self._description

    @property
    def encoder_cfg_key(self):
        return self.encoder_cfg.stem.upper()

    def iter_variants_params(self) -> Generator[Tuple[str, str], None, None]:
        for qp in self._variants:
            yield f'{self._anchor_key}-{qp}', str(qp)

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

    def get_metrics_set(self):
        assert self.reference is not None, 'AnchorTuple was loaded but reference sequence was either not specified, see AnchorTuple.load()'
        if self.reference.transfer_characteristics == TransferFunction.BT2020_PQ:
            return HDR_METRICS
        else:
            return SDR_METRICS


def ref_meta_location(row) -> str:
    loc = row[RefSequenceList.LOC]
    name = row[RefSequenceList.NAME]
    return f'{loc}/{name}.json'


def iter_ref_sequence_locations(reference_list: Path) -> Iterable[str]:
    refs = []
    with open(reference_list, 'r', encoding=ENCODING) as fo:
        for row in csv.DictReader(fo):
            key = row[RefSequenceList.KEY]
            loc = ref_meta_location(row)
            refs.append((key, loc))
    return refs


def reference_sequences_dict(reference_list: Path, ref_sequences_dir: Path, raises=False) -> Dict[str, VideoSequence]:
    """produces a dict mapping a reference sequence key
        to a VideoSequence if the sequence exists - None if it doesn't exist, or raises is True.
    """
    refs = {}
    with open(reference_list, 'r', encoding=ENCODING) as fo:
        for row in csv.DictReader(fo):
            meta = ref_sequences_dir / ref_meta_location(row)
            k = row[RefSequenceList.KEY]
            if not meta.exists():
                if raises:
                    raise FileNotFoundError(str(meta.resolve()))
                refs[k] = VideoSequence(meta, sequence={'Key': k})
                continue
            vs = VideoSequence.from_sidecar_metadata(meta)
            if vs.sequence:
                vs.sequence['Key'] = k
            refs[k] = vs
    return refs



def iter_anchors(streams_csv: Path, streams_dir: Path = None, sequences: Dict[str, VideoSequence] = None, cfg_dir=None, keys: Iterable[str] = None, cfg_keys: Iterable[str] = None) -> Iterable[AnchorTuple]:
    
    anchors = []

    if streams_dir is None:
        streams_dir = streams_csv.parent

    with open(streams_csv, 'r', encoding=ENCODING) as fo:
        for row in csv.DictReader(fo):
            anchor_key = row[AnchorList.KEY]
            if (keys is not None) and (anchor_key not in keys):
                continue
            if sequences is not None:
                sequence = sequences[row[AnchorList.REF_SEQ]]
                assert sequence is not None, f'[{anchor_key}] reference sequence not found: {row[AnchorList.REF_SEQ]}'
            else:
                sequence = None
            description = row[AnchorList.CLAUSE]
            encoder_id = row[AnchorList.REF_ENC] if AnchorList.REF_ENC in row else row[AnchorList.TEST_ENC]  # eg. HM16.22
            encoder_cfg_key = str(row[AnchorList.CFG])
            encoder_cfg = encoder_cfg_key.lower() + '.cfg'  # eg. S3-HM-01, no directory context specified
            if (cfg_keys is not None) and (encoder_cfg_key not in cfg_keys):
                continue
            if cfg_dir:
                encoder_cfg = Path(cfg_dir / encoder_cfg).resolve()
            variants = row[AnchorList.VARIANTS]
            anchor_dir = streams_dir / anchor_key
            # bitsream_key_template = row[AnchorList.VARIANT_KEY]
            anchors.append(
                AnchorTuple(anchor_dir, sequence, encoder_id, encoder_cfg, variants, anchor_key, description))
    
    return anchors


def load_variants(a: AnchorTuple, anchor_dir: Path = None) -> Iterable[Tuple[Path, VariantData]]:
    """Iterate the anchor's variant bitstream json metadata path,
        loads and returns VariantData if the metadata file exists.
    """
    if anchor_dir is None:
        anchor_dir = a.working_dir 
    assert anchor_dir.is_dir(), f'Directory not found: {anchor_dir}'
    for variant_id, qp in a.iter_variants_params():
        vfp = anchor_dir / f'{variant_id}.json'
        data = None
        if vfp.exists():
            data = VariantData.load(vfp, variant_id)
        yield qp, data
