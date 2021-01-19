from nn_models import EKETriNet
model = EKETriNet(in_features = 2, depth=6)
print(model.name)
print(model)
from torchsummary import summary
summary(model, (2,))

