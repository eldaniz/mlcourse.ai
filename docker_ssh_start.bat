rem docker exec -it %1 /bin/bash

docker exec -it aego service ssh restart
docker exec -it aego python3 -m spyder_kernels.console