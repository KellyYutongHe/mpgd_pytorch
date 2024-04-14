newgrp docker
docker build -t mpgd .
./run_docker_container.share
docker exec -it <name> bash