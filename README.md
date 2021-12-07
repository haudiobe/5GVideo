
## Quickstart

Get the code & build all docker images:
```
git clone https://github.com/haudiobe/5GVideo
cd 5GVideo
./build_all.sh
```

Set the environment variable pointing to your local data dir and start the docker-compose stack:
```
export VCC_WORKING_DIR=/path/to/local/data/5GVideo/
docker-compose up -d
```

At this point you should be able to monitor running tasks at:
[http://localhost:8888](http://localhost:8888)

to display get the tasks' logs:
```
docker-compose logs -f --tail 100 workers
```

now you should be able to run the scripts:
```
docker-compose exec python3 vcc.py --help
```
see the instructions below on scripts usage.



take down the docker-compose stack to stop and discard all running and pending tasks:
```
docker-compose down
```



## Downloading reference content

reference content is hosted at https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/

```
python3 ./download.py --help
python3 ./download.py streams --help
python3 ./download.py sequences --help
```

download anchor/test bitstreams
```
python3 download.py --pool-size 5 --verbose streams https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/Bitstreams/Scenario-1-FHD/264/streams.csv /data/Bitstreams/Scenario-1-FHD/264
```

download reference sequences
```
python3 download.py --pool-size 5 --verbose sequences /data/Bitstreams/Scenario-1-FHD/reference-sequence.csv /data/ReferenceSequences
```


## Encode, decode, convert, compute metrics

```
python3 ./src/vcc.py --help
```

typical commands to produce test/anchor data:

**Encode** all anchor variants for S1-A01-265:
```
python3 ./vcc.py -s S1-A01-265 encode
```

**Convert** the reference sequence and all anchor variants for S1-A01-265:
```
python3 ./vcc.py -s S1-A01-265 --sequence convert
python3 ./vcc.py -s S1-A01-265 --variants convert
```

**Compute** metrics for anchor variants for S1-A01-265:
```
python3 ./vcc.py -s S1-A01-265 metrics
```

All logs are stored in S1-A01-265, all dynamic parameters appear explicitely in the logs.


## Verifying reference results against local results

```
Usage: verify.py [OPTIONS] DOWNLOADED LOCAL COMMAND [ARGS]...

Options:
  --template FILE
  --help           Show this message and exit.

Commands:
  bitstream  verify that bistream md5 are matching
  decoder    verify that reconstruction md5 and metrics are matching
```
- *verification.csv* is created in *ANCHOR_STREAMS* parent directory


**bitstream verification**
```
export VCC_WORKING_DIR=/path/to
python3 download.py https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/Bitstreams/Scenario-1-FHD/264/streams.csv
python3 vcc.py decode
python3 vcc.py convert
python3 vcc.py metrics
python3 verify.py /path/to/Bitstreams/Scenario-1-FHD/264/streams.csv /path/to/Verification/Scenario-1-FHD/264/streams.csv bitstreams
```

**decoder verification**
```
python3 verify.py Bitstreams/Scenario-1-FHD/264/streams.csv BitstreamsVerification/Scenario-1-FHD/264/streams.csv decoder
```

## Metrics

```
python3 metrics.py --help
python3 metrics.py csv_metrics
```


## RD-plot and BD-rate computation

```
python3 compare.py --help
```


## running the scripts localy

python 3.8 or later is required (older versions not tested)

It is a standard python pratice to use a virtual envicronment to isolate dependencies:


**Linux/Mac/WSL**
```
python3 -m venv ./venv
source ./venv/bin/activate
```

**install dependencies** 
```
# source ./venv/bin/activate

python -m install -r ./src/requirements.txt
```

**now we can use the scripts** 
```
# source ./venv/bin/activate

python download.py --help
python vcc.py --help
python verify.py --help
```


## Environment variables

the scripts assumes the following environment variables:

- `VCC_WORKING_DIR` 

path to a local directory organized following the structure found at [https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/](https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/).


All other environment variables are not required if not used.

when using [JM]()
- `JM19_0_ENCODER`
- `JM19_0_DECODER`

absolute path to JM v19.0 encoder/decoder binaries and default configuration files

when using [HM]()
- `HM16_22_ENCODER` 
- `HM16_22_DECODER`
- `HM16_23_ENCODER`
- `HM16_23_DECODER`
- `SCM8_8_ENCODER`
- `SCM8_8_DECODER`
- `HM16_24_ENCODER`
- `HM16_24_DECODER`
- `HM_SEI_REMOVAL_APP`

absolute path to HM encoder and decoder binaries for each version used to generate anchors.



when using [VTM]()
- `VTM_13_0_ENCODER`
- `VTM_13_0_DECODER`
- `VTM_11_0_ENCODER`
- `VTM_11_0_DECODER`
- `SEI_REMOVAL_APP`

absolute path to VTM encoder and decoder binaries for each version used to generate anchors.


when using [ETM]()
- `ETM_7_5_ENCODER`
- `ETM_7_5_DECODER`

absolute path to ETM encoder and decoder binaries for each version used to generate anchors.


when using [HDRMetrics]()s
- `HDRMETRICS_TOOL`
- `HDRCONVERT_TOOL`
absolute path to HDMetrics executable

- `HDRMETRICS_CFG_DIR`
a directory containing the HDRTools config files found in this repo under **./docker/cfg/HDRTools**




## using docker

### build docker images

build all dependencies once:
`./build_all.sh`

this includes HDRTools, VMAF, all reference encoder, and produces 
