#!/bin/bash

ROOT_DIR="$(git rev-parse --show-toplevel)"
DOCKER_COMPOSE_DIR="$ROOT_DIR/.docker/motion_planning"
DOCKER_COMPOSE_FILE="$DOCKER_COMPOSE_DIR/docker-compose-motion_planning.yaml"
LAPTOP_IP="192.168.1.11"

echo "Welcome to the lite6_ws setup process."

# install docker
echo -e "\nInstall docker \n"

apt-get update
apt-get install ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# ensure GUI window is accessible from container
echo -e "Set Docker Xauth for x11 forwarding \n"

export DOCKER_XAUTH=/tmp/.docker.xauth
echo "export DOCKER_XAUTH=/tmp/.docker.xauth" >> ~/.bashrc
rm $DOCKER_XAUTH
touch $DOCKER_XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $DOCKER_XAUTH nmerge -

# find ethernet interface on device
echo -e "\n set static ip \n"

echo "Select an Ethernet interface to set a static IP for:"

interfaces=$(ip -o link show | grep -Eo '^[0-9]+: (en|eth|ens|eno|enp)[a-z0-9]*' | awk -F' ' '{print $2}')

# Display available interfaces for the user to choose from
select interface_name in $interfaces; do
    if [ -n "$interface_name" ]; then
        break
    else
        echo "Invalid selection. Please choose a valid interface."
    fi
done

echo "You've selected: $interface_name"

# Add and configure the static IP connection
nmcli connection delete "laptop_static"
nmcli connection add con-name "laptop_static" ifname "$interface_name" type ethernet
nmcli connection modify "laptop_static" ipv4.method manual ipv4.address $LAPTOP_IP/24
nmcli connection up "laptop_static"

echo "Static IP configuration complete for interface $interface_name."

