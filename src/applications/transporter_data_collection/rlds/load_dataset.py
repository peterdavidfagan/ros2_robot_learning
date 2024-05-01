from transporter import Transporter
import tensorflow_datasets as tfds

ds = tfds.load('transporter')['train']
for e in ds:
    try:
        print(e)
        print("eps")
    except:
        continue
print("test")