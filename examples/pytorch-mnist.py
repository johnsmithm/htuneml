from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from htuneml import Job

def getA():
    parser = argparse.ArgumentParser(description='Test a network')
    parser.add_argument('-n', '--name', type=str, help='name experiment', default='test')
    parser.add_argument('-e', '--epochs', type=int, help='number epochs', default=1)
    parser.add_argument('-d', '--debug', type=int, help='if 1 debug/do not sent experiment', default=0)
    parser.add_argument('-w', '--wait', type=int, help='if 1 wait for task from web app', default=0)
    parser.add_argument('-l', '--hidden', type=int, help='number neurons hidden layer', default=512)
    args = parser.parse_args()    
    return args

job = Job('apikey')

class Net(nn.Module):
    def __init__(self, hidden):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
@job.monitor
def main(pars):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=30, shuffle=True, **kwargs)
    model = Net(pars['hidden']).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(1, pars['epochs'] + 1):    
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        job.log({'ep':epoch, 'loss':loss.item()})

if __name__ == '__main__':
    args = getA()
    pars = vars(args)
    print(pars)
    
    if pars['debug'] == 1:
        job.debug()
    else:
        job.setName(pars['name'])
        
    if pars['wait'] == 1:
        job.sentParams(main)
        job.waitTask(main)
    else:
        main(pars)
        