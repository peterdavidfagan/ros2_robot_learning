# Camera Calibration

This application is deployed via Docker containers that are run on the Intel NUC and client machine. 

To run the control server run:

```
docker compose -f deploy_nuc.yaml up
```

To run the camera calibration app, cameras and motion planning software run:

```
docker compose -f deploy_client.yaml up
```
