from nn_models import EKEResnet
features = 5
model = EKEResnet(features)
print(model)
from torchsummary import summary
summary(model, (features,))

