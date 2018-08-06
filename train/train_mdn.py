from datasets.invert_sin import InvertedSineDataset
from models.mdn import MDN
from torch.utils.data import DataLoader
from torch.optim import RMSprop
from utils.args import get_args
from utils import device
from utils.helpers import mdn_loss_func

class MDNSineTrainer:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.args = get_args()
        self.loss_func = mdn_loss_func
        self.optimizer = RMSprop(self.model.parameters())

    def train(self):
        dataloader = DataLoader(self.dataset)

        for ep in range(self.args.epochs):

            for it, data in enumerate(dataloader):
                x_data, y_data = data
                res = self.model(x_data)

                loss = self.loss_func(*res, y_data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if it % self.args.log_every == 0:
                    print("ep: %d, it: %d, loss: %.4f" % (ep, it, loss))

if __name__ == '__main__':
    dataset = InvertedSineDataset(1000)
    model = MDN(1, 20, 5).to(device)
    trainer = MDNSineTrainer(dataset, model)
    trainer.train()

