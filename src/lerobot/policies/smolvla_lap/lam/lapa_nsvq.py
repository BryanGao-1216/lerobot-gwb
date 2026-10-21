# Adapted for LeRobot LAP: package-local, frozen inference only; see THIRD_PARTY.md.
"""LAPA NSVQ nearest-code inference; see LICENSE-LAPA.

Upstream NSVQ attribution: https://github.com/MHVali/Noise-Substitution-in-Vector-Quantization
NSVQ: Noise Substitution in Vector Quantization, IEEE Access, January 2022.
"""
import torch
import torch.distributions.uniform as uniform_dist

class NSVQ(torch.nn.Module):

    def __init__(self, dim, num_embeddings, embedding_dim, device=torch.device('cpu'), discarding_threshold=0.1, initialization='normal', code_seq_len=1, patch_size=32, image_size = 256):
        super(NSVQ, self).__init__()

        """
        Inputs:

        1. num_embeddings = Number of codebook entries

        2. embedding_dim = Embedding dimension (dimensionality of each input data sample or codebook entry)

        3. device = The device which executes the code (CPU or GPU)

        ########## change the following inputs based on your application ##########

        4. discarding_threshold = Percentage threshold for discarding unused codebooks

        5. initialization = Initial distribution for codebooks

        """
        self.image_size = image_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.discarding_threshold = discarding_threshold
        self.eps = 1e-12
        self.dim = dim
        self.patch_size = patch_size

        if initialization == 'normal':
            codebooks = torch.randn(self.num_embeddings, self.embedding_dim, device=device)
        elif initialization == 'uniform':
            codebooks = uniform_dist.Uniform(-1 / self.num_embeddings, 1 / self.num_embeddings).sample([self.num_embeddings, self.embedding_dim])
        else:
            raise ValueError("initialization should be one of the 'normal' and 'uniform' strings")

        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True)

        # Counter variable which contains the number of times each codebook is used
        self.codebooks_used = torch.zeros(self.num_embeddings, dtype=torch.int32, device=device)

        self.project_in = torch.nn.Linear(dim, embedding_dim)
        self.project_out = torch.nn.Linear(embedding_dim, dim)

        # 8 * 8  => 4 * 4 => 2 * 2
        #assert patch_size == 32
        if code_seq_len == 1:
            self.cnn_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=4, stride=1, padding=0),
            )
        elif code_seq_len == 2:
            self.cnn_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=(3, 4), stride=1, padding=0),
            )

        elif code_seq_len == 4:
            self.cnn_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=1, padding=0),
            )
        elif code_seq_len == 16:
            self.cnn_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4
            )
        elif code_seq_len == 64:
            self.cnn_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            )
        elif code_seq_len == 256:
            self.cnn_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            )
        else:
            raise ValueError("Not Implement: code_seq_len should be one of the 1 and 4 integers")

    def encode(self, input_data, batch_size):
        # compute the distances between input and codebooks vectors
        input_data = self.project_in(input_data) # b * 64 * 32
        # change the order of the input_data to b * 32 * 64
        input_data = input_data.permute(0, 2, 1).contiguous()
        # reshape input_data to 4D b*h*w*d
        input_data = input_data.reshape(batch_size, self.embedding_dim, int(self.image_size/self.patch_size), int(self.image_size/self.patch_size))
        input_data = self.cnn_encoder(input_data) # 1*1 tensor
        input_data = input_data.reshape(batch_size, self.embedding_dim, -1) # b * 32 * d^2
        input_data = input_data.permute(0, 2, 1).contiguous() # b * 1 * 32
        input_data = input_data.reshape(-1, self.embedding_dim)
        return input_data

    def decode(self, quantized_input, batch_size):
        quantized_input = quantized_input.reshape(batch_size, self.embedding_dim, -1) # b * 32 * d^2
        quantized_input = quantized_input.permute(0, 2, 1).contiguous() # b * 64 * 32

        quantized_input = self.project_out(quantized_input)
        return quantized_input

    def inference(self, input_data_first, input_data_last, user_action_token_num=None):

        """
        This function performs the vector quantization function for inference (evaluation) time (after training).
        This function should not be used during training.

        N: number of input data samples
        K: num_embeddings (number of codebook entries)
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        input: input_data (input data matrix which is going to be vector quantized | shape: (NxD) )
        outputs:
                quantized_input (vector quantized version of input data used for inference (evaluation) | shape: (NxD) )
        """

        #input_data = self.project_in(input_data)
        input_data_first = input_data_first.detach().clone()
        input_data_last = input_data_last.detach().clone()
        codebooks = self.codebooks.detach().clone()
        ###########################################

        batch_size = input_data_first.shape[0]
        # compute the distances between input and codebooks vectors

        input_data_first = self.encode(input_data_first, batch_size) # b * n * dim
        input_data_last = self.encode(input_data_last, batch_size) # b * n * dim

        input_data = input_data_last - input_data_first


        input_data = input_data.reshape(-1, self.embedding_dim)

        distances = (torch.sum(input_data ** 2, dim=1, keepdim=True)
                     - 2 * (torch.matmul(input_data, codebooks.t()))
                     + torch.sum(codebooks.t() ** 2, dim=0, keepdim=True))

        min_indices = torch.argmin(distances, dim=1)

        if user_action_token_num is not None:
            if type(user_action_token_num) == list:
                min_indices = torch.tensor(user_action_token_num, device=self.device)
            else:
                min_indices = torch.tensor([[user_action_token_num]], device=self.device).repeat(input_data.shape[0], 1)
        quantized_input = codebooks[min_indices]

        quantized_input = self.decode(quantized_input, batch_size)

        #use the tensor "quantized_input" as vector quantized version of your input data for inference (evaluation) phase.
        return quantized_input, min_indices.reshape(batch_size, -1)
