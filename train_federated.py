import random

import numpy as np
import torch
from tqdm import tqdm

from dataloader import MovielensDatasetLoader
from server_model import ServerNeuralCollaborativeFiltering
from train_single import NCFTrainer
from utils import Utils, seed_everything, plot_progress


class FederatedNCF:
    def __init__(self,
                 data_loader: MovielensDatasetLoader,
                 num_clients=50,
                 user_per_client_range=(1, 5),
                 mode="ncf",
                 aggregation_epochs=50,
                 local_epochs=10,
                 batch_size=128,
                 latent_dim=32,
                 seed=0):
        self.seed = seed
        seed_everything(seed)
        self.data_loader = data_loader
        self.ui_matrix = self.data_loader.ratings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_clients = num_clients
        self.latent_dim = latent_dim
        self.user_per_client_range = user_per_client_range
        self.mode = mode
        self.aggregation_epochs = aggregation_epochs
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.clients = self.generate_clients()
        self.ncf_optimizers = [torch.optim.Adam(client.ncf.parameters(), lr=0.001) for client in self.clients]
        self.utils = Utils(self.num_clients)
        self.hrs = []
        self.ndcg = []
        self.loss = []

    def generate_clients(self):
        start_index = 0
        clients = []
        for i in range(self.num_clients):
            users = random.randint(self.user_per_client_range[0], self.user_per_client_range[1])
            clients.append(NCFTrainer(user_ids=list(range(start_index, start_index + users)),
                                      data_loader=self.data_loader,
                                      epochs=self.local_epochs,
                                      batch_size=self.batch_size,
                                      latent_dim=self.latent_dim))
            start_index += users
        return clients

    def single_round(self, epoch=0):
        single_round_results = {key: [] for key in ["num_users", "loss", "hit_ratio@10", "ndcg@10"]}
        bar = tqdm(enumerate(self.clients), total=self.num_clients)
        for client_id, client in bar:
            results = client.train(self.ncf_optimizers[client_id])
            for k, i in results.items():
                single_round_results[k].append(i)
            printing_single_round = {"epoch": epoch}
            printing_single_round.update({k: round(sum(i) / len(i), 4) for k, i in single_round_results.items()})
            model = torch.jit.script(client.ncf.to(torch.device("cpu")))
            torch.jit.save(model, "./models/local/dp" + str(client_id) + ".pt")
            bar.set_description(str(printing_single_round))
        self.hrs.append(single_round_results["hit_ratio@10"])
        self.loss.append(single_round_results["loss"])
        self.ndcg.append(single_round_results["ndcg@10"])
        bar.close()

    def extract_item_models(self):
        for client_id in range(self.num_clients):
            model = torch.jit.load("./models/local/dp" + str(client_id) + ".pt")
            item_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1],
                                                            predictive_factor=self.latent_dim)
            item_model.set_weights(model)
            item_model = torch.jit.script(item_model.to(torch.device("cpu")))
            torch.jit.save(item_model, "./models/local_items/dp" + str(client_id) + ".pt")

    def train(self):
        server_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1],
                                                          predictive_factor=self.latent_dim)
        server_model = torch.jit.script(server_model.to(torch.device("cpu")))
        torch.jit.save(server_model, "./models/central/server" + str(0) + ".pt")
        for epoch in range(self.aggregation_epochs):
            server_model = torch.jit.load("./models/central/server" + str(epoch) + ".pt",
                                          map_location=self.device)
            _ = [client.ncf.to(self.device) for client in self.clients]
            _ = [client.ncf.load_server_weights(server_model) for client in self.clients]
            self.single_round(epoch=epoch)
            self.extract_item_models()
            self.utils.federate()

        progress = {
            "hit_ratio@10": np.mean(self.hrs, axis=1),
            "loss": np.mean(self.loss, axis=1),
            "ndcg@10": np.mean(self.ndcg, axis=1)
        }
        plot_progress(progress, title=f"120 Client, Seed = {self.seed}", loss="MSE")


if __name__ == '__main__':
    dataloader = MovielensDatasetLoader(thresh=1)
    seeds = {117623077}  # , 117623077}  # , 204110176}
    # 187372311, 129995678,
    # 6155814, 22612812, 61168821,
    # 21228945, 146764631, 94412880,
    # }
    for s in seeds:
        fncf = FederatedNCF(
            data_loader=dataloader,
            num_clients=120,
            user_per_client_range=[1, 10],
            mode="ncf",
            aggregation_epochs=400,
            local_epochs=2,
            batch_size=128,
            latent_dim=12,
            seed=s
        )
        fncf.train()
