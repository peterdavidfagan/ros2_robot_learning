# Camera Calibration

https://github.com/peterdavidfagan/moveit2_camera_calibration/assets/42982057/9c4e3532-9f56-4a14-89c3-fb88c366c038

This application is deployed via Docker containers that are run on the Intel NUC and client machine. 

To run the control server run:

```
docker compose -f deploy_nuc.yaml up
```

To run the camera calibration app, cameras and motion planning software run:

```
docker compose -f deploy_client.yaml up
```
