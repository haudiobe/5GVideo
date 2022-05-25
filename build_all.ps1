
if (-not (Test-Path 'env:VCC_WORKING_DIR')) { 
    echo 'VCC_WORKING_DIR is not set'
    exit 1
}

docker build -t vcc:base -f ./docker/Dockerfile.base ./docker
docker build -t vcc:hm -f ./docker/Dockerfile.HM ./docker
docker build -t vcc:jm -f ./docker/Dockerfile.JM ./docker
docker build -t vcc:vtm -f ./docker/Dockerfile.VTM ./docker
docker build -t vcc:etm -f ./docker/Dockerfile.ETM ./docker
docker build -t vcc:aom -f ./docker/Dockerfile.AOM ./docker
docker build -t vcc:worker -f ./docker/Dockerfile.worker ./
docker-compose build