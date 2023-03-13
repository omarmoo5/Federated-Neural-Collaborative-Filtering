import torch


class ServerNeuralCollaborativeFiltering(torch.nn.Module):
    def __init__(self, item_num, predictive_factor=12):
        super(ServerNeuralCollaborativeFiltering, self).__init__()
        self.mlp_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=predictive_factor)
        self.gmf_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=predictive_factor)
        self.gmf_out = torch.nn.Linear(predictive_factor, 1)
        self.gmf_out.weight = torch.nn.Parameter(torch.ones(1, predictive_factor))
        # Linear layers of MLP model that we will pass the concatenation of the user and item latent vectors to
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * predictive_factor, 48), torch.nn.ReLU(),
            torch.nn.Linear(48, 24), torch.nn.ReLU(),
            torch.nn.Linear(24, 12), torch.nn.ReLU(),
            torch.nn.Linear(12, 6), torch.nn.ReLU()
        )
        self.mlp_out = torch.nn.Linear(6, 1)

        self.output_logits = torch.nn.Linear(predictive_factor, 1)
        self.model_blending = 0.5  # alpha parameter, equation 13 in the paper
        self.initialize_weights()
        self.join_output_weights()

    def initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.mlp_item_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.gmf_item_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.gmf_out.weight)
        torch.nn.init.xavier_uniform_(self.mlp_out.weight)
        torch.nn.init.xavier_uniform_(self.output_logits.weight)

    def layer_setter(self, model, model_copy):
        for m, mc in zip(model.parameters(), model_copy.parameters()):
            mc.data[:] = m.data[:]

    def set_weights(self, model):
        self.layer_setter(model.mlp_item_embeddings, self.mlp_item_embeddings)
        self.layer_setter(model.gmf_item_embeddings, self.gmf_item_embeddings)
        self.layer_setter(model.mlp, self.mlp)
        self.layer_setter(model.gmf_out, self.gmf_out)
        self.layer_setter(model.mlp_out, self.mlp_out)
        self.layer_setter(model.output_logits, self.output_logits)

    def forward(self):
        return torch.tensor(0.0)

    def join_output_weights(self):
        W = torch.nn.Parameter(
            torch.cat((self.model_blending * self.gmf_out.weight, (1 - self.model_blending) * self.mlp_out.weight),
                      dim=1))
        self.output_logits.weight = W


if __name__ == '__main__':
    ncf = ServerNeuralCollaborativeFiltering(100, 64)
    print(ncf)
