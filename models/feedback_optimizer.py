import torch

class DirectFeedbackAlignmentOptimizer(torch.optim.Optimizer):
    """Custom optimizer for feedback alignment.
    """
    def __init__(self, params, layer_sizes, device, lr=0.001, feedback_noise_scale=1, use_random_noise=False):
        if lr <= 0.0:
            raise ValueError("Learning rate must be positive.")
        defaults = dict(lr=lr)
        self.num_layers = len(layer_sizes) - 1
        super(DirectFeedbackAlignmentOptimizer, self).__init__(params, defaults)

        output_size = layer_sizes[-1]
        if use_random_noise:
            # Initialise fixed random feedback matrices from output to each hidden layer
            self.feedback_matrices = [
                torch.randn(output_size, layer_sizes[i]).to(device) * feedback_noise_scale
                for i in range(1, self.num_layers)]
        else:
            # Extract weights from the model parameters
            weight_matrices = []
            for param_idx, param in enumerate(self.param_groups[0]['params']):
                if len(param.shape) == 2:  # Only consider weight matrices, not biases
                    weight_matrices.append(param.detach().clone())

            # Initialise feedback matrices using matrix multiplication of model weights
            self.feedback_matrices = []
            for i in range(1, self.num_layers):
                if i == self.num_layers - 1:
                    # For the last hidden layer, use the output layer weights directly
                    feedback = weight_matrices[-1].clone()
                else:
                    # For earlier layers, multiply weights of all subsequent layers
                    feedback = weight_matrices[-1].clone()
                    for j in range(len(weight_matrices) - 2, i - 1, -1):
                        feedback = feedback @ weight_matrices[j]

                # Apply noise scaling if needed
                feedback = feedback * feedback_noise_scale
                self.feedback_matrices.append(feedback)

    @torch.no_grad()
    def step(self, loss, output, inputs, activations):
        """
            Perform one update step using DFA.
        """
        # print('output shape: ', output.shape)
        # print('input shapes: ', [i.shape for i in inputs])
        # print('activation shapes: ', [i.shape for i in activations])
        # Get the output error signal
        output_error = torch.autograd.grad(loss, output, retain_graph=True)[0]
        # print('output_error shape: ', output_error.shape)
        for group in self.param_groups:
            lr = group['lr']
            for param_idx, param in enumerate(group['params']):
                if len(param.shape) == 2:
                    layer_idx = param_idx // 2
                    # print(f'weight at layer {layer_idx} shape: ', param.shape)
                    if layer_idx == self.num_layers-1:
                        # Last layer: use true gradient
                        grad = inputs[layer_idx].T @ output_error
                    else:
                        # Hidden layers: use direct feedback
                        # print('feedback_matrix shape: ', self.feedback_matrices[layer_idx].shape)
                        feedback_signal = output_error @ self.feedback_matrices[layer_idx]
                        # print('feedback_signal shape: ', feedback_signal.shape)
                        delta = feedback_signal * activations[layer_idx]
                        # print('delta shape: ', delta.shape)
                        grad = inputs[layer_idx].T @ delta
                        # print('grad shape: ', grad.T.shape)
                    # Apply the weight update
                    param.sub_(lr * grad.T)
                elif len(param.shape) == 1:
                    layer_idx = (param_idx - 1) // 2
                    # print(f'bias at layer {layer_idx} shape: ', param.shape)
                    if layer_idx == self.num_layers - 1:
                        # Last layer: use true gradient
                        grad = output_error.sum(dim=0)
                    else:
                        # Hidden layers: use direct feedback
                        feedback_signal = output_error @ self.feedback_matrices[layer_idx]
                        delta = feedback_signal * activations[layer_idx]
                        grad = delta.sum(dim=0)

                        # Apply the bias update
                    param.sub_(lr * grad)