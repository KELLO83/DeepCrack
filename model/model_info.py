import torchinfo
from deepcrack import DeepCrack


model = DeepCrack()

torchinfo.summary(model , input_size=(1,3,512,512))