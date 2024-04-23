#!/bin/bash

# Define the IP addresses of yur machines
CLIENT_IP="192.168.106.20"
CLIENT_HOSTNAME="r2d2"
NUC_IP="192.168.106.10"
NUC_HOSTNAME="franka-control"

# Initialize Docker Swarm on the manager node
docker swarm init --advertise-addr $CLIENT_IP

# Extract the join token
JOIN_TOKEN=$(docker swarm join-token -q worker)

# Join worker nodes to the Swarm
ssh franka-control@$NUC_IP "docker swarm join --token $JOIN_TOKEN $CLIENT_IP:2377"

# apply node labels for deployment
docker node update --label-add type=client $(docker node ls -qf "name=$CLIENT_HOSTNAME")
docker node update --label-add type=nuc $(docker node ls -qf "name=$NUC_HOSTNAME")

# Verify Swarm status
docker node ls
