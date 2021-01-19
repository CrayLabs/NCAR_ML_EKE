from nn_models import EKEResnet
features = 11
model = EKEResnet(features)
print(model)
from torchsummary import summary
summary(model, (features,))

