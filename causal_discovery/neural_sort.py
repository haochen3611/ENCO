import torch
import torch.nn.functional as F
import numpy as np
import math
import random
import sys

sys.path.append("../")

from causal_graphs.variable_distributions import _random_categ
from causal_discovery.datasets import InterventionalDataset


class NeuralSorting(object):

    def __init__(
        self,
        model,
        graph,
        num_batches,
        num_graphs,
        batch_size,
        sample_size_inters,
        max_graph_stacking=200,
        exclude_inters=None,
    ):
        """
        Creates a DistributionFitting object that summarizes all functionalities
        for performing the graph fitting stage of ENCO.

        Parameters
        ----------
        model : MultivarMLP
                PyTorch module of the neural networks that model the conditional
                distributions.
        graph : CausalDAG
                Causal graph on which we want to perform causal structure learning.
        num_batches : int
                      Number of batches to use per MC sample in the graph fitting stage.
                      Usually 1, only higher needed if GPU is running out of memory for
                      common batch sizes.
        num_graphs : int
                     Number of graph samples to use for estimating the gradients in the
                     graph fitting stage. Usually in the range 20-100.
        theta_only_num_graphs : int
                                Number of graph samples to use in the graph fitting stage if
                                gamma is frozen. Needs to be an even number, and usually 2 or 4.
        batch_size : int
                     Size of the batches to use in the gradient estimators.
        lambda_sparse : float
                        Sparsity regularizer value to use in the graph fitting stage.
        sample_size_inters: Number of samples to use per intervention. If an exported graph is
                            given as input and sample_size_inters is smaller than the exported
                            interventional dataset, the first sample_size_inters samples will be taken.
        max_graph_stacking : int
                             Number of graphs that can maximally evaluated in parallel on the device.
                             If you run out of GPU memory, try to lower this number. It will then
                             evaluate the graph sequentially, which can be slightly slower but uses
                             less memory.
        exclude_inters : list
                         A list of variable indices that should be excluded from sampling interventions
                         from. This should be used to apply ENCO on intervention sets on a subset of
                         the variable set. If None, an empty list will be assumed, i.e., interventions
                         on all variables will be used.
        """
        self.model = model
        self.graph = graph
        self.tau = None
        self.beta = None
        self.num_batches = num_batches
        self.num_graphs = num_graphs
        self.sample_size_inters = sample_size_inters
        self.batch_size = batch_size
        self.max_graph_stacking = max_graph_stacking
        self.inter_vars = []
        self.exclude_inters = exclude_inters if exclude_inters is not None else list()
        self.theta_grad_mask = torch.zeros(self.graph.num_vars, self.graph.num_vars)
        for v in self.exclude_inters:
            self.theta_grad_mask[v, self.exclude_inters] = 1.0
        self.dataset = InterventionalDataset(
            self.graph, dataset_size=self.sample_size_inters, batch_size=self.batch_size
        )
        if len(self.exclude_inters) > 0:
            print(
                f"Excluding interventions on the following {len(self.exclude_inters)}"
                f" out of {graph.num_vars} variables: "
                f'{", ".join([str(i) for i in sorted(self.exclude_inters)])}'
            )

    def perform_update_step(self, gamma, tau=0.1, beta=1.0, var_idx=-1):
        """
        Performs a full update step of the graph fitting stage. We first sample a batch of graphs,
        evaluate them on a interventional data batch, and estimate the gradients for gamma and theta
        based on the log-likelihoods.

        Parameters
        ----------
        gamma : nn.Parameter
                Parameter tensor representing the gamma parameters in ENCO.
        theta : nn.Parameter
                Parameter tensor representing the theta parameters in ENCO.
        var_idx : int
                  Variable on which should be intervened to obtain the update. If none is given, i.e.,
                  a negative value, the variable will be randomly selected.
        only_theta : bool
                     If True, gamma is frozen and the gradients are only estimated for theta. See
                     Appendix D.2 in the paper for details on the gamma freezing stage.
        """
        # Obtain log-likelihood estimates for randomly sampled graph structures

        self.tau = tau
        self.beta = beta

        gamma.requires_grad_(True)

        sampled_order, log_likelihoods, var_idx = self.get_monte_carlo_samples(
            gamma, self.num_batches, self.num_graphs, self.batch_size, var_idx
        )

        # Determine gradients for gamma and theta
        gamma_grad = self.gradient_estimator(
            sampled_order, log_likelihoods, gamma, var_idx)
        
        log_likelihoods = log_likelihoods.detach()
        log_likelihoods[var_idx] = 0.
        loss = -log_likelihoods.sum()

        return loss, var_idx

    def get_monte_carlo_samples(
        self, gamma, num_batches, num_graphs, batch_size, var_idx=-1
    ):

        device = self.get_device()

        # Sample data batch
        if hasattr(self, "dataset"):
            # Pre-sampled data
            var_idx = self.sample_next_var_idx()
            int_sample = torch.cat(
                [self.dataset.get_batch(var_idx) for _ in range(num_batches)], dim=0
            ).to(device)
            batch_size = int_sample.shape[0] // num_batches
        else:
            # If no dataset exists, data is newly sampled from the graph
            intervention_dict, var_idx = self.sample_intervention(
                self.graph, dataset_size=num_batches * batch_size, var_idx=var_idx
            )
            int_sample = self.graph.sample(
                interventions=intervention_dict,
                batch_size=num_batches * batch_size,
                as_array=True,
            )
            int_sample = torch.from_numpy(int_sample).to(device)

        num_graphs_list = [
            min(self.max_graph_stacking, num_graphs - i * self.max_graph_stacking)
            for i in range(math.ceil(num_graphs * 1.0 / self.max_graph_stacking))
        ]
        num_graphs_list = [
            (num_graphs_list[i], sum(num_graphs_list[:i]))
            for i in range(len(num_graphs_list))
        ]

        sampled_perm = self.gumbel_softmax(
            num_graphs, gamma, tau=self.tau, beta=self.beta
        )
        lower_triangluar = (
            torch.tril(sampled_perm.size(-1), 0)
            .unsqueeze(0)
            .expand(sampled_perm.size(0), -1, -1)
        )
        adj_matrix = torch.enisum("bij,bjk->bik", sampled_perm, lower_triangluar)

        # Evaluate log-likelihoods under sampled adjacency matrix and data
        log_likelihoods = []
        for n_idx in range(num_batches):
            batch = int_sample[n_idx * batch_size : (n_idx + 1) * batch_size]

            for c_idx, (graph_count, start_idx) in enumerate(num_graphs_list):
                adj_matrix_expanded = (
                    adj_matrix[start_idx : start_idx + graph_count, None]
                    .expand(-1, batch_size, -1, -1)
                    .flatten(0, 1)
                )
                batch_exp = batch[None, :].expand(graph_count, -1, -1).flatten(0, 1)
                nll = self.evaluate_likelihoods(batch_exp, adj_matrix_expanded, var_idx)
                nll = nll.reshape(graph_count, batch_size, -1)

                if n_idx == 0:
                    log_likelihoods.append(nll.mean(dim=1))
                else:
                    log_likelihoods[c_idx] += nll.mean(dim=1)

        # Combine all data
        # adj_matrices = torch.cat(adj_matrices, dim=0)
        log_likelihoods = torch.cat(log_likelihoods, dim=0).mean(dim=0)

        sampled_order = sampled_perm.argmax(dim=-1)

        return sampled_order, log_likelihoods, var_idx

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
            log_s_perturb = torch.log(scores) + sample_gumbel(
                [n_samples, n_dim, 1], beta=beta
            ).to(scores.device)
            P_hat = self.neural_sort(log_s_perturb, tau=tau)
            P_hat = P_hat.view(-1, n_dim, n_dim)  # n_samples x n_dim x n_dim

        return P_hat

    def neural_sort(self, scores, tau=1.0, hard=False):
        """Performs the neural sort operation on the given scores.

        Args:
            scores (Tensor): the parameter tensor representing the scores.
            tau (float, optional): temperature parameter. Defaults to 1.0.
            hard (bool, optional): use hard sort or not. Defaults to False.

        Returns:
            Tensor: the permuation matrix.
        """

        n_batch = scores.size(0)
        n_dim = scores.size(1)
        ones = torch.ones(n_batch, n_dim, 1, dtype=torch.float, device=scores.device)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))

        B = torch.matmul(A_scores, torch.matmul(ones, torch.transpose(ones, 0, 1)))

        scaling = torch.matmul(n_dim + 1 - 2 * (torch.arange(n_dim) + 1)).float()

        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C - B).permute(0, 2, 1)
        sm = torch.nn.Softmax(dim=-1)
        P_hat = sm(P_max / tau)

        if hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            b_idx = (
                torch.arange(n_batch)
                .repeat([1, n_dim])
                .view(n_dim, n_batch)
                .transpose(1, 0)
                .flatten()
                .type(torch.long)
            )
            r_idx = torch.arange(n_dim).repeat([n_batch, 1]).flatten().type(torch.long)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P - P_hat).detach() + P_hat

        return P_hat

    @torch.no_grad()
    def gradient_estimator(self, sampled_order, log_likelihoods, gamma, var_idx, use_auto_grad=True):
        """
        Returns the estimated gradients for gamma and theta. It uses the low-variance gradient estimators
        proposed in Section 3.3 of the paper.

        Parameters
        ----------
        adj_matrices : torch.FloatTensor, shape [batch_size, num_vars, num_vars]
                       The adjacency matrices on which the interventional data has been evaluated on.
        log_likelihoods : torch.FloatTensor, shape [num_vars, ]
                          The average log-likelihood under the adjacency matrices for all variables
                          in the graph.
        gamma : nn.Parameter
                Parameter tensor representing the gamma parameters in ENCO.
        theta : nn.Parameter
                Parameter tensor representing the theta parameters in ENCO.
        var_idx : int
                  Variable on which the intervention was performed.
        """

        batch_size = sampled_order.shape[0]

        if use_auto_grad:

            log_likelihoods.backward(torch.ones_like(log_likelihoods))

            gamma.grad[var_idx, :] = 0.
            gamma.grad = gamma.grad.sum(dim=0)

            return gamma.grad
        
        else:

            pass


    def sample_next_var_idx(self):
        """
        Returns next variable to intervene on. We iterate through the variables
        in a shuffled order, like a standard dataset.
        """
        if len(self.inter_vars) == 0:  # If an epoch finished, reshuffle variables
            self.inter_vars = [
                i
                for i in range(len(self.graph.variables))
                if i not in self.exclude_inters
            ]
            random.shuffle(self.inter_vars)
        var_idx = self.inter_vars.pop()
        return var_idx

    def sample_intervention(self, graph, dataset_size, var_idx=-1):
        """
        Returns a new data batch for an intervened variable.
        """
        # Select variable to intervene on
        if var_idx < 0:
            var_idx = self.sample_next_var_idx()
        var = graph.variables[var_idx]
        # Soft, perfect intervention => replace p(X_n) by random categorical
        # Scale is set to 0.0, which represents a uniform distribution.
        int_dist = _random_categ(size=(var.prob_dist.num_categs,), scale=0.0, axis=-1)
        # Sample from interventional distribution
        value = np.random.multinomial(n=1, pvals=int_dist, size=(dataset_size,))
        value = np.argmax(value, axis=-1)  # One-hot to index
        intervention_dict = {var.name: value}

        return intervention_dict, var_idx

    @torch.no_grad()
    def evaluate_likelihoods(self, int_sample, adj_matrix, var_idx):
        """
        Evaluates the negative log-likelihood of the interventional data batch (int_sample)
        on the given graph structures (adj_matrix) and the intervened variable (var_idx).
        """
        self.model.eval()
        device = self.get_device()
        int_sample = int_sample.to(device)
        adj_matrix = adj_matrix.to(device)
        # Transpose for mask because adj[i,j] means that i->j
        mask_adj_matrix = adj_matrix.transpose(1, 2)
        preds = self.model(int_sample, mask=mask_adj_matrix)

        # Evaluate negative log-likelihood of predictions
        if int_sample.dtype == torch.long:
            preds = preds.flatten(0, 1)
            labels = int_sample.clone()
            labels[:, var_idx] = (
                -1
            )  # Perfect interventions => no predictions of the intervened variable
            labels = labels.reshape(-1)
            nll = F.cross_entropy(preds, labels, reduction="none", ignore_index=-1)
            nll = nll.reshape(*int_sample.shape)
        else:
            nll = preds

        self.model.train()
        return nll

    def get_device(self):
        return self.model.device
