#!/usr/bin/env bash
# run_parallel.sh
docker-compose up -d sbx1 sbx2 sbx3
docker-compose logs -f sbx1 sbx2 sbx3
docker-compose down sbx1 sbx2 sbx3
docker-compose up --no-deps --build sbx4