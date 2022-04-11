
## Metrics verification

This guide provides step by step instructions to process with verification of anchor reconstructions and verifications.


### Download content

1. download bitstreams and jsons for a given scenario

```
python3 src/download.py streams https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/Bitstreams/Scenario-1-FHD/VTM/streams.csv /data/Bitstreams/Scenario-1-FHD/VTM


/data
 └ Bitstreams
  └ Scenario-1-FHD
    |   reference-sequence.csv
    └ VTM
     |   streams.csv
     └── CFG
     └── Metrics
     └── S1-T11-VTM
     |   └── S1-T11-VTM-22
     |       S1-T11-VTM-22.json
     |       S1-T11-VTM-22.bin
     |   └── S1-T11-VTM-27
     |   └── S1-T11-VTM-32
     |   └── S1-T11-VTM-37
    [...]
     └ S1-T17-VTM
```


2. download reference sequences for that scenario

```
python3 src/download.py sequences /data/Bitstreams/Scenario-1-FHD/reference-sequence.csv /data/ReferenceSequences


/data
  └ ReferenceSequences
    └ Life-Untouched
        Life-Untouched-FHD.json
        Life-Untouched-FHD.yuv
```



### Generate reconstruction of bitstreams

reconstruct QP 22 for S1-T11-VTM
```
python3 src/vcc.py decode -s S1-T11-VTM-22 decode


/data/Bitstreams/Scenario-1-FHD
 └── S1-T11-VTM
     └── S1-T11-VTM-22
         S1-T11-VTM-22.yuv
         S1-T11-VTM-22.dec.log
```


### Format conversion before metrics computation

In some cases, must process conversion from one representation to another before.
This 8 to 10 bit, 10 to 8 bit, and YUV to RGB444.

In the case of S1-T11-VTM, YUV to RGB444 needs to be performed on the reference sequence as well as on the reconstructions to compute HDR metrics using HDRTools.


Convert the reconstructions of the test scenario:

```
python3 src/vcc.py -s S1-T11-VTM-22 convert --reconstructions


/data/Bitstreams/Scenario-1-FHD
 └── S1-T11-VTM
     └── S1-T11-VTM-22
         |   S1-T11-VTM-22.yuv
         |   S1-T11-VTM-22.yuv.json
         └── tmp
                S1-T11-VTM-22_2020_444_%05d.json
                S1-T11-VTM-22_2020_444_000000.exr
                S1-T11-VTM-22_2020_444_000000.exr
                S1-T11-VTM-22_2020_444_000001.exr
                ...
                S1-T11-VTM-22_2020_444_%05d.hdrconvert.log
     ...
```


Convert the reference sequence used to encode S1-T11-VTM:

```
python3 src/vcc.py -s S1-T11-VTM convert --sequences


/data/ReferenceSequences
 └── Life-Untouched
     |   Life-Untouched-FHD.yuv
     |   Life-Untouched-FHD.json
     └── tmp
            Life-Untouched-FHD_2020_444_%05d.json
            Life-Untouched-FHD_2020_444_000000.exr
            Life-Untouched-FHD_2020_444_000001.exr
            ...
            Life-Untouched-FHD_2020_444_%05d.hdrconvert.log
 ...
```


### Compute metrics

Run metrics computations, and update the json metadata. 

```
python3 src/vcc.py -s S1-T11-VTM-22 metrics

/data/Bitstreams/Scenario-1-FHD
 └── S1-T11-VTM
     └── S1-T11-VTM-22
         |   S1-T11-VTM-22.metrics.log
         |   S1-T11-VTM-22.json
         |   ...
         └── tmp
                S1-T11-VTM-22_2020_444_%05d.metrics.log
                ...
```

Using --dry-run skips computation but attempts to parse an existing log to update the .json metadata.






### Export csv metrics

```
python3 src/vcc.py -s S1-T11-VTM-22 csv-metrics
```


### Download the original metrics

In order to produce a verification report, we must download the original metrics.

```
python3 src/vcc.py -c S1-VTM-02 csv-metrics
```


### Generate verification report

```
python3 src/vcc.py -c S1-VTM-02 convert --reconstructions
```

