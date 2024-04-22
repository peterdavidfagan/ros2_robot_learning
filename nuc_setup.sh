#!/bin/bash

export ROOT_DIR=$(git rev-parse --show-toplevel)
export NUC_IP="192.168.1.10"

echo "Welcome to the lite6_ws setup process."

read -p "Is this your first time setting up the machine? (yes/no): " first_time

if [ "$first_time" = "yes" ]; then
        echo "Great! Let's proceed with the setup."

        # ensure submodules are cloned
        echo "Repulling all submodules."
        read -p "Enter the user whose ssh credentials will be used: " USERNAME
        eval "$(ssh-agent -s)"
        ssh-add /home/$USERNAME/.ssh/id_ed25519
        ROOT_DIR="$(git rev-parse --show-toplevel)"
        cd $ROOT_DIR && git submodule update --recursive --init

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
        systemctl enable docker

        # perform rt-patch
        echo -e "\nPerform realtime patch of kernel \n"

        apt update && apt install ubuntu-advantage-tools
        pro attach $UBUNTU_PRO_TOKEN
        pro enable realtime-kernel

        # cpu frequency scaling
        echo -e "\nSet cpu frequency scaling settings \n"

        apt install cpufrequtils -y
        systemctl disable ondemand
        systemctl enable cpufrequtils
        sh -c 'echo "GOVERNOR=performance" > /etc/default/cpufrequtils'
        systemctl daemon-reload && sudo systemctl restart cpufrequtils

else
    echo -e "\nWelcome back!\n"
fi

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
nmcli connection delete "nuc_static"
nmcli connection add con-name "nuc_static" ifname "$interface_name" type ethernet
nmcli connection modify "nuc_static" ipv4.method manual ipv4.address $NUC_IP/24 
nmcli connection up "nuc_static"

echo "Static IP configuration complete for interface $interface_name."

# turn on ssh
echo -e "\n turn on ssh \n"
systemctl enable ssh

# turn off display manager
systemctl disable display-manager.service

