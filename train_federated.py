import torch
from train_single import NCFTrainer
from dataloader import MovielensDatasetLoader
import random
from tqdm import tqdm
from server_model import ServerNeuralCollaborativeFiltering
import copy


class Utils:
    def __init__(self, num_clients, local_path="./models/local_items/", server_path="./models/central/"):
        self.epoch = 0
        self.num_clients = num_clients
        self.local_path = local_path
        self.server_path = server_path

    def load_pytorch_client_model(self, path):
        #Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
        return torch.jit.load(path)

    def get_user_models(self, loader):
        #get clients models from "./models/local_items/"
        models = []
        for client_id in range(self.num_clients):
            models.append({'model': loader(self.local_path + "dp" + str(client_id) + ".pt")})
        return models

    def get_previous_federated_model(self):
        self.epoch += 1
        return torch.jit.load(self.server_path + "server" + str(self.epoch - 1) + ".pt")

    def save_federated_model(self, model):

        torch.jit.save(model, self.server_path + "server" + str(self.epoch) + ".pt")

    #after each epoch the framework saves the result in "models/server"


def federate(utils):
    client_models = utils.get_user_models(utils.load_pytorch_client_model)
    server_model = utils.get_previous_federated_model() #get the last model saved by the last epoch
    if len(client_models) == 0:
        utils.save_federated_model(server_model) #if there's no models for clients then save the last readed model
        return
    n = len(client_models)

    #deep copy :a copy of the object is copied into another object.
    # It means that any changes made to a copy of the object do not reflect in the original object.
    # Access the model and optimizer state_dict


    #model 'mlp_item_embeddings.weight', 'gmf_item_embeddings.weight', 'mlp.0.weight'
    # , 'mlp.0.bias', 'mlp.2.weight', 'mlp.2.bias' , 'mlp.4.weight', 'mlp.4.bias'
    # , 'gmf_out.weight', 'gmf_out.bias', 'mlp_out.weight', 'mlp_out.bias', 'output_logits.weight', 'output_logits.bias'

    # agg function (server weights+client weights)/#clients
    server_new_dict = copy.deepcopy(client_models[0]['model'].state_dict())
    for i in range(1, len(client_models)):
        client_dict = client_models[i]['model'].state_dict()
        for k in client_dict.keys():
            server_new_dict[k] += client_dict[k]
    for k in server_new_dict.keys():
        server_new_dict[k] = server_new_dict[k] / n
    print(server_new_dict.keys())
    server_model.load_state_dict(server_new_dict)
    utils.save_federated_model(server_model)


class FederatedNCF:
    def __init__(self, ui_matrix, num_clients=50, user_per_client_range=[1, 5], mode="ncf", aggregation_epochs=50,
                 local_epochs=10, batch_size=128, latent_dim=32, seed=0):
        random.seed(seed)
        self.ui_matrix = ui_matrix
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_clients = num_clients
        self.latent_dim = latent_dim
        self.user_per_client_range = user_per_client_range
        self.mode = mode
        self.aggregation_epochs = aggregation_epochs
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.clients = self.generate_clients()
        self.ncf_optimizers = [torch.optim.Adam(client.ncf.parameters(), lr=5e-4) for client in self.clients]
        self.utils = Utils(self.num_clients)

    def generate_clients(self):
        start_index = 0
        clients = []
        for i in range(self.num_clients):
            users = random.randint(self.user_per_client_range[0], self.user_per_client_range[1])
            clients.append(NCFTrainer(self.ui_matrix[start_index:start_index + users], epochs=self.local_epochs,
                                      batch_size=self.batch_size))
            start_index += users
        return clients

    def single_round(self, epoch=0, first_time=False):
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
        first_time = True
        server_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1],
                                                          predictive_factor=self.latent_dim)
        server_model = torch.jit.script(server_model.to(torch.device("cpu")))
        torch.jit.save(server_model, "./models/central/server" + str(0) + ".pt")
        for epoch in range(self.aggregation_epochs):
            server_model = torch.jit.load("./models/central/server" + str(epoch) + ".pt", map_location=self.device)
            o = [client.ncf.to(self.device) for client in self.clients]
            p = [client.ncf.load_server_weights(server_model) for client in self.clients]
            # print(o)
            # print(p)
            self.single_round(epoch=epoch, first_time=first_time)
            first_time = False
            self.extract_item_models()
            federate(self.utils)


if __name__ == '__main__':
    dataloader = MovielensDatasetLoader()
    fncf = FederatedNCF(dataloader.ratings, num_clients=5, user_per_client_range=[1, 5], mode="ncf",
                        aggregation_epochs=5, local_epochs=10, batch_size=128)
    fncf.train()
