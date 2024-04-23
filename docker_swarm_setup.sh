#!/bin/bash

# Define the IP addresses of yur machines
CLIENT_IP="192.168.106.20"
NUC_IP="192.168.106.10"

# Initialize Docker Swarm on the manager node
docker swarm init --advertise-addr $CLIENT_IP

# Extract the join token
JOIN_TOKEN=$(docker swarm join-token -q worker)

# Join worker nodes to the Swarm
ssh franka-control@$NUC_IP "docker swarm join --token $JOIN_TOKEN $CLIENT_IP:2377"

# Verify Swarm status
docker node ls
