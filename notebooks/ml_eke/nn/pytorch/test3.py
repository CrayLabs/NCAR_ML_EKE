from nn_models import EKEResnetSmall
features = 5
model = EKEResnetSmall(features)
print(model)
from torchsummary import summary
summary(model, (features,))

