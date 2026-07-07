import sys

sys.path.append('../Activity5/')
import torch
from save_model import Classifier

checkpoint = torch.load("../Activity5/checkpoint.pth")
layers = [100, 100, 100, 50]

model = Classifier(checkpoint["input"], layers)

model.load_state_dict(checkpoint['state_dict'])

x_test = torch.tensor([[0.0606, 0.5, 0.333, 0.4828, 0.4000, 0.4000, 0.4000, 0.4000, 0.4000, \
        0.4000, 0.1651, 0.0869, 0.980, 0.1825, 0.1054, 0.2807, 0.0016, 0.000, 0.0033, \
        0.0027, 0.0031, 0.0021]]).float()

# Making a prediction
pred = model(x_test)
pred_exp = torch.exp(pred)
top_p, top_class_test = pred_exp.topk(1, dim=1)

if top_class_test == 1:
    print(f'With the probability of {pred_exp}, the client is likely to default')
else:
    print(f'With the probability of {pred_exp}, the client is likely to default')

# Convert into jit
traced_script = torch.jit.trace(model, x_test, check_trace=False)
prediction = traced_script(x_test)
prediction_exp = torch.exp(prediction)
top_p, top_class_test = prediction_exp.topk(1, dim=1)

if top_class_test == 1:
    print(f'With the probability of {pred_exp}, the client is likely to default')
else:
    print(f'With the probability of {pred_exp}, the client is likely to default')
