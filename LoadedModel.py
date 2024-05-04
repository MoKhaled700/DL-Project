import torch.nn
from Model import ConvNet
from data import CustomData
import torch.utils.data

PATH = 'Resources/cnn.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test = CustomData('Resources/driving_log.csv')

loader = torch.utils.data.DataLoader(test, 32, shuffle=False)
loaded_model = ConvNet()
loaded_model.load_state_dict(torch.load(PATH)) # it takes the loaded dictionary, not the path file itself
loaded_model.to(device)
loaded_model.eval()

loaded_model = ConvNet()
loaded_model.load_state_dict(torch.load(PATH)) # it takes the loaded dictionary, not the path file itself
loaded_model.to(device)
loaded_model.eval()

with torch.no_grad():
    n_correct = 0
    n_correct2 = 0
    n_samples = len(loader.dataset)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = loaded_model(images)

        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

        outputs2 = loaded_model(images)
        _, predicted2 = torch.max(outputs2, 1)
        n_correct2 += (predicted2 == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the model: {acc} %')

    acc = 100.0 * n_correct2 / n_samples
    print(f'Accuracy of the loaded model: {acc} %')

