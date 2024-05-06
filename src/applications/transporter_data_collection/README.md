[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Zu_bj0xQGbxMKLTK0X1SR-WrK71Iv5-B?usp=sharing)

# Transporter Data Collection

https://github.com/peterdavidfagan/moveit2_data_collector/assets/42982057/f2a3e4ac-b8eb-466c-b17f-bdd2d8fa688b

This application is deployed via Docker containers that are run on the Intel NUC and client machine. 

To run the control server run:

```
docker compose -f deploy_nuc.yaml up
```

To run the camera calibration app, cameras and motion planning software run:

```
docker compose -f deploy_client.yaml up
```

# Published Datasets

An example dataset which was collected using this software can be found [here](https://huggingface.co/datasets/peterdavidfagan/transporter_networks).
