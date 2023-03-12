import numpy as np
import torch

from dataloader import MovielensDatasetLoader
from metrics import compute_metrics
from model import NeuralCollaborativeFiltering
from utils import plot_progress, seed_everything


class MatrixLoader:
    def __init__(self,
                 ui_matrix,
                 dataloader: MovielensDatasetLoader = None,
                 default=None,
                 user_ids=None):
        self.ui_matrix = ui_matrix
        self.positives = np.argwhere(self.ui_matrix != 0)
        self.negatives = np.argwhere(self.ui_matrix == 0)
        for i, usr_id in enumerate(user_ids):
            self.positives[self.positives[:, 0] == i, 0] = usr_id
            self.negatives[self.negatives[:, 0] == i, 0] = usr_id
        self.user_ids = user_ids
        self.dataloader = dataloader
        if user_ids:
            latest_interactions = np.array(
                [[usr_id, dataloader.latest_ratings[usr_id]["item_id"]] for usr_id in user_ids])
            mask = np.array([not np.array_equal(row, rows_to_remove_i) for row in self.positives for rows_to_remove_i in
                             latest_interactions]).reshape(self.positives.shape[0], latest_interactions.shape[0]).all(
                axis=1)
            self.positives = self.positives[mask]
        if default is None:
            self.default = np.array([[0, 0]]), np.array([0])
        else:
            self.default = default

    def delete_indexes(self, indexes, arr="pos"):
        if arr == "pos":
            self.positives = np.delete(self.positives, indexes, 0)
        else:
            self.negatives = np.delete(self.negatives, indexes, 0)

    def get_batch(self, batch_size):
        if self.positives.shape[0] < batch_size // 4 or self.negatives.shape[0] < batch_size - batch_size // 4:
            return torch.tensor(self.default[0]), torch.tensor(self.default[1])
        try:
            pos_indexes = np.random.choice(self.positives.shape[0], batch_size // 4)
            neg_indexes = np.random.choice(self.negatives.shape[0], batch_size - batch_size // 4)
            pos = self.positives[pos_indexes]
            neg = self.negatives[neg_indexes]
            self.delete_indexes(pos_indexes, "pos")
            self.delete_indexes(neg_indexes, "neg")
            batch = np.concatenate((pos, neg), axis=0)
            if batch.shape[0] != batch_size:
                return torch.tensor(self.default[0]), torch.tensor(self.default[1]).float()
            np.random.shuffle(batch)
            y = np.array([self.dataloader.ratings[i][j] for i, j in batch])
            return torch.tensor(batch), torch.tensor(y).float()
        except Exception as exp:
            print(exp)
            return torch.tensor(self.default[0]), torch.tensor(self.default[1]).float()

    def get_test_batch(self):
        user_id = np.random.choice(self.user_ids)
        pos = np.array([user_id, self.dataloader.latest_ratings[user_id]["item_id"]])
        neg_indexes = np.random.choice(self.negatives.shape[0], 99)
        neg = self.negatives[neg_indexes]
        batch = np.concatenate((pos.reshape(1, -1), neg), axis=0)
        return torch.tensor(batch)


class NCFTrainer:
    def __init__(self,
                 data_loader: MovielensDatasetLoader,
                 user_ids,
                 epochs,
                 batch_size,
                 latent_dim=32,
                 device=None):
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.loader = None
        self.user_ids = user_ids
        self.data_loader = data_loader
        self.ui_matrix = self.data_loader.get_ui_matrix(self.user_ids)
        self.initialize_loader()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ncf = NeuralCollaborativeFiltering(self.data_loader.ratings.shape[0], self.data_loader.ratings.shape[1],
                                                self.latent_dim).to(self.device)

    def initialize_loader(self):
        self.loader = MatrixLoader(self.ui_matrix, dataloader=self.data_loader, user_ids=self.user_ids)

    def train_batch(self, x, y, optimizer):
        self.ncf.train()  # set the model to training mode
        optimizer.zero_grad()  # zero the gradients
        y_ = self.ncf(x)  # forward pass
        loss = torch.nn.functional.binary_cross_entropy(y_, y)  # compute the loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), y_.detach()

    def train_model(self, optimizer, epochs=None, print_num=10):
        epoch = 0
        progress = {"epoch": [], "loss": [], "hit_ratio@10": [], "ndcg@10": []}
        running_loss, running_hr, running_ndcg = 0, 0, 0
        prev_running_loss, prev_running_hr, prev_running_ndcg = 0, 0, 0
        if epochs is None:
            epochs = self.epochs
        steps, prev_steps, prev_epoch = 0, 0, 0
        while epoch < epochs:
            x, y = self.loader.get_batch(self.batch_size)
            if x.shape[0] < self.batch_size:
                prev_running_loss, prev_running_hr, prev_running_ndcg = running_loss, running_hr, running_ndcg
                running_loss = 0
                running_hr = 0
                running_ndcg = 0
                prev_steps = steps
                steps = 0
                epoch += 1
                self.initialize_loader()
                x, y = self.loader.get_batch(self.batch_size)
            x, y = x.int(), y.float()
            x, y = x.to(self.device), y.to(self.device)
            loss, y_ = self.train_batch(x, y, optimizer)
            test_batch = self.loader.get_test_batch()
            hr, ndcg = compute_metrics(model=self.ncf, test_batch=test_batch, device=self.device)
            running_loss += loss
            running_hr += hr
            running_ndcg += ndcg
            if epoch != 0 and steps == 0:
                results = {"epoch": prev_epoch, "loss": prev_running_loss / (prev_steps + 1),
                           "hit_ratio@10": prev_running_hr / (prev_steps + 1),
                           "ndcg@10": prev_running_ndcg / (prev_steps + 1)}
            else:
                results = {"epoch": prev_epoch, "loss": running_loss / (steps + 1),
                           "hit_ratio@10": running_hr / (steps + 1), "ndcg@10": running_ndcg / (steps + 1)}
            steps += 1
            if prev_epoch != epoch:
                progress["epoch"].append(results["epoch"])
                progress["loss"].append(results["loss"])
                progress["hit_ratio@10"].append(results["hit_ratio@10"])
                progress["ndcg@10"].append(results["ndcg@10"])
                prev_epoch += 1
        r_results = {"num_users": self.ui_matrix.shape[0]}
        r_results.update({i: results[i] for i in ["loss", "hit_ratio@10", "ndcg@10"]})
        return r_results, progress

    def train(self, ncf_optimizer, return_progress=False):
        self.ncf.join_output_weights()
        results, progress = self.train_model(ncf_optimizer)
        if return_progress:
            return results, progress
        else:
            return results


if __name__ == '__main__':
    seed_everything(0)
    dataloader = MovielensDatasetLoader(thresh=1)
    users = 10
    trainer = NCFTrainer(epochs=20,
                         batch_size=128,
                         data_loader=dataloader,
                         user_ids=list(range(users)))
    ncf_optimizer = torch.optim.Adam(trainer.ncf.parameters(), lr=5e-4)
    _, progress = trainer.train(ncf_optimizer, return_progress=True)
    plot_progress(progress, f"{users} Users on 1 Client", "MSE")
