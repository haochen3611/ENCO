import torch
import torch.nn as nn


class DistributionFittingNS(object):

    def __init__(self, model, optimizer, data_loader):
        """
        Creates a DistributionFitting object that summarizes all functionalities
        for performing the distribution fitting stage of ENCO.

        Parameters
        ----------
        model : MultivarMLP
                PyTorch module of the neural networks that model the conditional
                distributions.
        optimizer : torch.optim.Optimizer
                    Standard PyTorch optimizer for the model.
        data_loader : torch.data.DataLoader
                      Data loader returning batches of observational data. This
                      data is used for training the neural networks.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_module = nn.CrossEntropyLoss()
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)

    def _get_next_batch(self):
        """
        Helper function for sampling batches one by one from the data loader.
        """
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)
        return batch

    def perform_update_step(self, gamma, tau=0.1, beta=1.0):
        """
        Performs a full update step of the distribution fitting stage.
        It first samples a batch of random adjacency matrices from 'sample_matrix',
        and then performs a training step on a random observational data batch.

        Parameters
        ----------
        sample_matrix : torch.FloatTensor, shape [num_vars, num_vars]
                        Float tensor with values between 0 and 1. An element (i,j)
                        represents the probability of having an edge from X_i to X_j,
                        i.e., not masking input X_i for predicting X_j.

        Returns
        -------
        loss : float
               The loss of the model with the sampled adjacency matrices on the
               observational data batch.
        """
        batch = self._get_next_batch()
        adj_matrices = self.sample_graphs(
            gamma=gamma,
            batch_size=batch.shape[0],
            tau=tau,
            beta=beta
        )
        loss = self.train_step(batch, adj_matrices)
        return loss

    @torch.no_grad()
    def sample_graphs(self, gamma, batch_size, tau, beta):
        """
        Samples a batch of adjacency matrices that are used for masking the inputs.
        """
        sampled_perm = self.gumbel_softmax(n_samples=batch_size, scores=gamma, tau=tau, beta=beta)
        ones_tril = torch.ones(sampled_perm.size(-1), sampled_perm.size(-1), device=sampled_perm.device, dtype=sampled_perm.dtype)
        ones_tril = torch.tril(ones_tril, 0).unsqueeze(0).expand(sampled_perm.size(0), -1, -1)
        adj_matrices = torch.einsum("bij,bjk->bik", sampled_perm, ones_tril)
        # Mask diagonals
        return adj_matrices

    def train_step(self, inputs, adj_matrices):
        """
        Performs single optimization step of the neural networks
        on given inputs and adjacency matrix.
        """
        self.model.train()
        self.optimizer.zero_grad()
        device = self.model.device
        inputs = inputs.to(device)
        adj_matrices = adj_matrices.to(device)
        ## Transpose for mask because adj[i,j] means that i->j
        # no need to transpose here
        # mask_adj_matrices = adj_matrices.transpose(1, 2)
        preds = self.model(inputs, mask=adj_matrices)

        if inputs.dtype == torch.long:
            loss = self.loss_module(preds.flatten(0,-2), inputs.reshape(-1))
        else:  # If False, our input was continuous, and we return log likelihoods as preds
            loss = preds.mean()

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def gumbel_softmax(self, n_samples, scores, tau=1.0, beta=1.0):
        """
        
        Args:
            n_samples: int
                Number of samples to draw.
            scores: torch.Tensor
                Scores to use for the Gumbel softmax. Shape: (n_dim, )
            tau: float
                Temperature parameter for the Gumbel softmax.
            beta: float
                Beta parameter for the Gumbel softmax.
        """

        n_dim = scores.shape[0]
        scores = scores.unsqueeze(0).unsqueeze(-1)

        def sample_gumbel(samples_shape, beta=1.0, eps=1e-20):
            U = torch.zeros(samples_shape).uniform_()
            return -torch.log(-torch.log(U + eps) + eps) * beta
        
        with torch.enable_grad():
            log_s_perturb = torch.log(scores) + sample_gumbel([n_samples, n_dim, 1], beta=beta).to(scores.device)
            P_hat = self.neural_sort(log_s_perturb, tau=tau)
            P_hat = P_hat.view(-1, n_dim, n_dim) # n_samples x n_dim x n_dim

        return P_hat
    
    def neural_sort(self, scores, tau=1.0, hard=False):
        
        n_batch = scores.size(0)
        n_dim = scores.size(1)
        ones = torch.ones(n_dim, 1, dtype=torch.float, device=scores.device)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))

        B = torch.matmul(A_scores, torch.matmul(ones, torch.transpose(ones, 0, 1)))

        scaling = (n_dim + 1 - 2 * (torch.arange(n_dim) + 1)).float().to(scores.device)

        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C - B).permute(0, 2, 1)
        sm = torch.nn.Softmax(dim=-1)
        P_hat = sm(P_max / tau)

        if hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            b_idx = torch.arange(n_batch).repeat([1, n_dim]).view(n_dim, n_batch).transpose(1, 0).flatten().type(torch.long)
            r_idx = torch.arange(n_dim).repeat([n_batch, 1]).flatten().type(torch.long)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P - P_hat).detach() + P_hat
        
        return P_hat