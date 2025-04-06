
import torch
import torch.nn as nn
import torch.nn.functional as F

class Skeletonize(torch.nn.Module):
    """
    Class based on PyTorch's Module class to skeletonize two- or three-dimensional input images
    while being fully compatible with PyTorch's autograd automatic differention engine as proposed in [1].

    Attributes:
        propabilistic: a Boolean that indicates whether the input image should be binarized using
                       the reparametrization trick and straight-through estimator.
                       It should always be set to True if non-binary inputs are being provided.
        beta: scale of added logistic noise during the reparametrization trick. If too small, there will not be any learning via
              gradient-based optimization; if too large, the learning is very slow.
        tau: Boltzmann temperature for reparametrization trick.
        simple_point_detection: decides whether simple points should be identified using Boolean characterization of their 26-neighborhood (Boolean) [2]
                                or by checking whether the Euler characteristic changes under their deletion (EulerCharacteristic) [3].
        num_iter: number of iterations that each include one end-point check, eight checks for simple points and eight subsequent deletions.
                  The number of iterations should be tuned to the type of input image.

    [1] Martin J. Menten et al. A skeletonization algorithm for gradient-based optimization.
        Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.
    [2] Gilles Bertrand. A boolean characterization of three- dimensional simple points.
        Pattern recognition letters, 17(2):115-124, 1996.
    [3] Steven Lobregt et al. Three-dimensional skeletonization:principle and algorithm.
        IEEE Transactions on pattern analysis and machine intelligence, 2(1):75-77, 1980.
    """

    def __init__(self, probabilistic=True, beta=0.33, tau=1.0, simple_point_detection='Boolean', num_iter=5):

        super(Skeletonize, self).__init__()
        
        self.probabilistic = probabilistic
        self.tau = tau
        self.beta = beta

        self.num_iter = num_iter
        self.endpoint_check = self._single_neighbor_check
        if simple_point_detection == 'Boolean':
            self.simple_check = self._boolean_simple_check
        elif simple_point_detection == 'EulerCharacteristic':
            self.simple_check = self._euler_characteristic_simple_check
        else:
            raise Exception()


    def forward(self, img):

        img = self._prepare_input(img)

        if self.probabilistic:
            img = self._stochastic_discretization(img)

        for current_iter in range(self.num_iter):

            # At each iteration create a new map of the end-points
            is_endpoint = self.endpoint_check(img)

            # Sub-iterate through eight different subfields
            x_offsets = [0, 1, 0, 1, 0, 1, 0, 1]
            y_offsets = [0, 0, 1, 1, 0, 0, 1, 1]
            z_offsets = [0, 0, 0, 0, 1, 1, 1, 1]

            for x_offset, y_offset, z_offset in zip(x_offsets, y_offsets, z_offsets):

                # At each sub-iteration detect all simple points and delete all simple points that are not end-points
                is_simple = self.simple_check(img[:, :, x_offset:, y_offset:, z_offset:])
                deletion_candidates = is_simple * (1 - is_endpoint[:, :, x_offset::2, y_offset::2, z_offset::2])
                img[:, :, x_offset::2, y_offset::2, z_offset::2] = torch.min(img[:, :, x_offset::2, y_offset::2, z_offset::2].clone(), 1 - deletion_candidates)

        img = self._prepare_output(img)

        return img



    def _prepare_input(self, img):
        """
        Function to check that the input image is compatible with the subsequent calculations.
        Only two- and three-dimensional images with values between 0 and 1 are supported.
        If the input image is two-dimensional then it is converted into a three-dimensional one for further processing.
        """

        if img.dim() == 5:
            self.expanded_dims = False
        elif img.dim() == 4:
            self.expanded_dims = True
            img = img.unsqueeze(2)
        else:
            raise Exception("Only two-or three-dimensional images (tensor dimensionality of 4 or 5) are supported as input.")

        if img.shape[2] == 2 or img.shape[3] == 2 or img.shape[4] == 2 or img.shape[3] == 1 or img.shape[4] == 1:
            raise Exception()

        if img.min() < 0.0 or img.max() > 1.0:
            raise Exception("Image values must lie between 0 and 1.")

        img = F.pad(img, (1, 1, 1, 1, 1, 1), value=0)
        
        return img


    def _stochastic_discretization(self, img):
        """
        Function to binarize the image so that it can be processed by our skeletonization method.
        In order to remain compatible with backpropagation we utilize the reparameterization trick and a straight-through estimator.
        """

        alpha = (img + 1e-8) / (1.0 - img + 1e-8)

        uniform_noise = torch.rand_like(img)
        uniform_noise = torch.empty_like(img).uniform_(1e-8, 1 - 1e-8)
        logistic_noise = (torch.log(uniform_noise) - torch.log(1 - uniform_noise))

        img = torch.sigmoid((torch.log(alpha) + logistic_noise * self.beta) / self.tau)
        img = (img.detach() > 0.5).float() - img.detach() + img

        return img


    def _single_neighbor_check(self, img):
        """
        Function that characterizes points as endpoints if they have a single neighbor or no neighbor at all.
        """

        img = F.pad(img, (1, 1, 1, 1, 1, 1))

        # Check that number of ones in twentysix-neighborhood is exactly 0 or 1
        K = torch.tensor([[[1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0]],
                          [[1.0, 1.0, 1.0],
                           [1.0, 0.0, 1.0],
                           [1.0, 1.0, 1.0]],
                          [[1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0]]], device=img.device).view(1, 1, 3, 3, 3)

        num_twentysix_neighbors = F.conv3d(img, K)
        condition1 = F.hardtanh(-(num_twentysix_neighbors - 2), min_val=0, max_val=1) # 1 or fewer neigbors
        
        return condition1


    def _boolean_simple_check(self, img):
        """
        Function that identifies simple points using Boolean conditions introduced by Bertrand et al. [1].
        Each Boolean conditions can be assessed via convolutions with a limited number of pre-defined kernels.
        It total, four conditions are checked. If any one is fulfilled, the point is deemed simple.

        [1] Gilles Bertrand. A boolean characterization of three- dimensional simple points.
            Pattern recognition letters, 17(2):115-124, 1996.
        """

        img = F.pad(img, (1, 1, 1, 1, 1, 1), value=0)

        # Condition 1: number of zeros in the six-neighborhood is exactly 1
        K_N6 = torch.tensor([[[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0]],
                             [[0.0, 1.0, 0.0],
                              [1.0, 0.0, 1.0],
                              [0.0, 1.0, 0.0]],
                             [[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        num_six_neighbors = F.conv3d(1 - img, K_N6, stride=2)

        subcondition1a = F.hardtanh(num_six_neighbors, min_val=0, max_val=1) # 1 or more neighbors
        subcondition1b = F.hardtanh(-(num_six_neighbors - 2), min_val=0, max_val=1) # 1 or fewer neighbors
        
        condition1 = subcondition1a * subcondition1b


        # Condition 2: number of ones in twentysix-neighborhood is exactly 1
        K_N26 = torch.tensor([[[1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0]],
                              [[1.0, 1.0, 1.0],
                               [1.0, 0.0, 1.0],
                               [1.0, 1.0, 1.0]],
                              [[1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0]]], device=img.device).view(1, 1, 3, 3, 3)

        num_twentysix_neighbors = F.conv3d(img, K_N26, stride=2)

        subcondition2a = F.hardtanh(num_twentysix_neighbors, min_val=0, max_val=1) # 1 or more neighbors
        subcondition2b = F.hardtanh(-(num_twentysix_neighbors - 2), min_val=0, max_val=1) # 1 or fewer neigbors
        
        condition2 =  subcondition2a * subcondition2b


        # Condition 3: Number of ones in eighteen-neigborhood exactly 1...
        K_N18 = torch.tensor([[[0.0, 1.0, 0.0],
                               [1.0, 1.0, 1.0],
                               [0.0, 1.0, 0.0]],
                              [[1.0, 1.0, 1.0],
                               [1.0, 0.0, 1.0],
                               [1.0, 1.0, 1.0]],
                              [[0.0, 1.0, 0.0],
                               [1.0, 1.0, 1.0],
                               [0.0, 1.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        num_eighteen_neighbors = F.conv3d(img, K_N18, stride=2)

        subcondition3a = F.hardtanh(num_eighteen_neighbors, min_val=0, max_val=1) # 1 or more neighbors
        subcondition3b = F.hardtanh(-(num_eighteen_neighbors - 2), min_val=0, max_val=1) # 1 or fewer neigbors

        # ... and cell configration B26 does not exist
        K_B26 =  torch.tensor([[[1.0, -1.0, 0.0],
                                [-1.0, -1.0, 0.0],
                                [0.0, 0.0, 0.0]],
                               [[-1.0, -1.0, 0.0],
                                [-1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]],
                               [[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        B26_1_present = F.relu(F.conv3d(2.0 * img - 1.0, K_B26, stride=2) - 6)
        B26_2_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[2]), stride=2) - 6)
        B26_3_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[3]), stride=2) - 6)
        B26_4_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[4]), stride=2) - 6)
        B26_5_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[2, 3]), stride=2) - 6)
        B26_6_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[2, 4]), stride=2) - 6)
        B26_7_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[3, 4]), stride=2) - 6)
        B26_8_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[2, 3, 4]), stride=2) - 6)
        num_B26_cells = B26_1_present + B26_2_present + B26_3_present + B26_4_present + B26_5_present + B26_6_present + B26_7_present + B26_8_present

        subcondition3cand4a = F.hardtanh(-(num_B26_cells - 1), min_val=0, max_val=1)

        condition3 = subcondition3a * subcondition3b


        # Condition 4: cell configuration A6 does not exist...
        K_A6 = torch.tensor([[[0.0, 1.0, 0.0],
                              [1.0, -1.0, 1.0],
                              [0.0, 1.0, 0.0]],
                             [[0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]],
                             [[0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        A6_1_present = F.relu(F.conv3d(2.0 * img - 1.0, K_A6, stride=2) - 4)
        A6_2_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_A6, dims=[2, 3]), stride=2) - 4)
        A6_3_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_A6, dims=[2, 4]), stride=2) - 4)
        A6_4_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_A6, dims=[2]), stride=2) - 4)
        A6_5_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(torch.flip(K_A6, dims=[2]), dims=[2, 3]), stride=2) - 4)
        A6_6_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(torch.flip(K_A6, dims=[2]), dims=[2, 4]), stride=2) - 4)
        num_A6_cells = A6_1_present + A6_2_present + A6_3_present + A6_4_present + A6_5_present + A6_6_present

        subcondition4a = F.hardtanh(-(num_A6_cells - 1), min_val=0, max_val=1)

        # ... and cell configuration B26 does not exist...
        # K_B26 =  torch.tensor([[[1.0, -1.0, 0.0],
        #                         [-1.0, -1.0, 0.0],
        #                         [0.0, 0.0, 0.0]],
        #                        [[-1.0, -1.0, 0.0],
        #                         [-1.0, 0.0, 0.0],
        #                         [0.0, 0.0, 0.0]],
        #                        [[0.0, 0.0, 0.0],
        #                         [0.0, 0.0, 0.0],
        #                         [0.0, 0.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        # B26_1_present = F.relu(F.conv3d(2.0 * img - 1.0, K_B26, stride=2) - 6)
        # B26_2_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[2]), stride=2) - 6)
        # B26_3_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[3]), stride=2) - 6)
        # B26_4_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[4]), stride=2) - 6)
        # B26_5_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[2, 3]), stride=2) - 6)
        # B26_6_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[2, 4]), stride=2) - 6)
        # B26_7_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[3, 4]), stride=2) - 6)
        # B26_8_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[2, 3, 4]), stride=2) - 6)
        # num_B26_cells = B26_1_present + B26_2_present + B26_3_present + B26_4_present + B26_5_present + B26_6_present + B26_7_present + B26_8_present

        # subcondition4b = F.hardtanh(-(num_B26_cells - 1), min_val=0, max_val=1)
        
        # ... and cell configuration B18 does not exist...
        K_B18 = torch.tensor([[[0.0, 1.0, 0.0],
                               [-1.0, -1.0, -1.0],
                               [0.0, 0.0, 0.0]],
                              [[-1.0, -1.0, -1.0],
                               [-1.0, 0.0, -1.0],
                               [0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        B18_1_present = F.relu(F.conv3d(2.0 * img - 1.0, K_B18, stride=2) - 8)
        B18_2_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_B18, dims=[2, 4]), stride=2) - 8)
        B18_3_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_B18, dims=[2, 4], k=2), stride=2) - 8)
        B18_4_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_B18, dims=[2, 4], k=3), stride=2) - 8)
        B18_5_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_B18, dims=[3, 4]), stride=2) - 8)
        B18_6_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_B18, dims=[3, 4]), dims=[2, 4]), stride=2) - 8)
        B18_7_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_B18, dims=[3, 4]), dims=[2, 4], k=2), stride=2) - 8)
        B18_8_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_B18, dims=[3, 4]), dims=[2, 4], k=3), stride=2) - 8)
        B18_9_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_B18, dims=[3, 4], k=2), stride=2) - 8)
        B18_10_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_B18, dims=[3, 4], k=2), dims=[2, 4]), stride=2) - 8)
        B18_11_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_B18, dims=[3, 4], k=2), dims=[2, 4], k=2), stride=2) - 8)
        B18_12_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_B18, dims=[3, 4], k=2), dims=[2, 4], k=3), stride=2) - 8)
        num_B18_cells = B18_1_present + B18_2_present + B18_3_present + B18_4_present + B18_5_present + B18_6_present + B18_7_present + B18_8_present + B18_9_present + B18_10_present + B18_11_present + B18_12_present

        subcondition4c = F.hardtanh(-(num_B18_cells - 1), min_val=0, max_val=1)

        # ... and the number of zeros in the six-neighborhood minus the number of A18 cell configurations plus the number of A26 cell configurations is exactly one
        K_N6 = torch.tensor([[[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0]],
                             [[0.0, 1.0, 0.0],
                              [1.0, 0.0, 1.0],
                              [0.0, 1.0, 0.0]],
                             [[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        num_six_neighbors = F.conv3d(1-img, K_N6, stride=2)

        K_A18 = torch.tensor([[[0.0, -1.0, 0.0],
                               [0.0, -1.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              [[0.0, -1.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        A18_1_present = F.relu(F.conv3d(2.0 * img - 1.0, K_A18, stride=2) - 2)
        A18_2_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_A18, dims=[2, 4]), stride=2) - 2)
        A18_3_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_A18, dims=[2, 4], k=2), stride=2) - 2)
        A18_4_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_A18, dims=[2, 4], k=3), stride=2) - 2)
        A18_5_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_A18, dims=[3, 4]), stride=2) - 2)
        A18_6_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_A18, dims=[3, 4]), dims=[2, 4]), stride=2) - 2)
        A18_7_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_A18, dims=[3, 4]), dims=[2, 4], k=2), stride=2) - 2)
        A18_8_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_A18, dims=[3, 4]), dims=[2, 4], k=3), stride=2) - 2)
        A18_9_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_A18, dims=[3, 4], k=2), stride=2) - 2)
        A18_10_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_A18, dims=[3, 4], k=2), dims=[2, 4]), stride=2) - 2)
        A18_11_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_A18, dims=[3, 4], k=2), dims=[2, 4], k=2), stride=2) - 2)
        A18_12_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_A18, dims=[3, 4], k=2), dims=[2, 4], k=3), stride=2) - 2)
        num_A18_cells = A18_1_present + A18_2_present + A18_3_present + A18_4_present + A18_5_present + A18_6_present + A18_7_present + A18_8_present + A18_9_present + A18_10_present + A18_11_present + A18_12_present

        K_A26 = torch.tensor([[[-1.0, -1.0, 0.0],
                               [-1.0, -1.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              [[-1.0, -1.0, 0.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        A26_1_present = F.relu(F.conv3d(2.0 * img - 1.0, K_A26, stride=2) - 6)
        A26_2_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_A26, dims=[2]), stride=2) - 6)
        A26_3_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_A26, dims=[3]), stride=2) - 6)
        A26_4_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_A26, dims=[4]), stride=2) - 6)
        A26_5_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_A26, dims=[2, 3]), stride=2) - 6)
        A26_6_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_A26, dims=[2, 4]), stride=2) - 6)
        A26_7_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_A26, dims=[3, 4]), stride=2) - 6)
        A26_8_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_A26, dims=[2, 3, 4]), stride=2) - 6)
        num_A26_cells = A26_1_present + A26_2_present + A26_3_present + A26_4_present + A26_5_present + A26_6_present + A26_7_present + A26_8_present

        subcondition4d = F.hardtanh(num_six_neighbors - num_A18_cells + num_A26_cells, min_val=0, max_val=1) # 1 or more configurations
        subcondition4e = F.hardtanh(-(num_six_neighbors - num_A18_cells + num_A26_cells - 2), min_val=0, max_val=1) # 1 or fewer configurations

        condition4 = subcondition4a * subcondition4c * subcondition4d * subcondition4e

        # We can do B26 && (X18 || A6 && B18 && (X6 - A18 + A26))
        condition34 = subcondition3cand4a * torch.cat([condition3, condition4], dim=1)

        # If any of the four conditions is fulfilled the point is simple
        combined = torch.cat([condition1, condition2, condition34], dim=1)
        is_simple = torch.amax(combined, dim=1, keepdim=True)
        
        return is_simple


    # Specifically designed to be used with the eight-subfield iterative scheme from above.
    def _euler_characteristic_simple_check(self, img):
        """
        Function that identifies simple points by assessing whether the Euler characteristic changes when deleting it [1].
        In order to calculate the Euler characteristic, the amount of vertices, edges, faces and octants are counted using convolutions with pre-defined kernels.
        The function is meant to be used in combination with the subfield-based iterative scheme employed in the forward function.

        [1] Steven Lobregt et al. Three-dimensional skeletonization:principle and algorithm.
            IEEE Transactions on pattern analysis and machine intelligence, 2(1):75-77, 1980.
        """

        img = F.pad(img, (1, 1, 1, 1, 1, 1), value=0)

        # Create masked version of the image where the center of 26-neighborhoods is changed to zero
        mask = torch.ones_like(img)
        mask[:, :, 1::2, 1::2, 1::2] = 0
        masked_img = img.clone() * mask

        # Count vertices
        vertices = F.relu(-(2.0 * img - 1.0))
        num_vertices = F.avg_pool3d(vertices, (3, 3, 3), stride=2) * 27

        masked_vertices = F.relu(-(2.0 * masked_img - 1.0))
        num_masked_vertices = F.avg_pool3d(masked_vertices, (3, 3, 3), stride=2) * 27
    
        # Count edges
        K_ud_edge = torch.tensor([0.5, 0.5], device=img.device).view(1, 1, 2, 1, 1)
        K_ns_edge = torch.tensor([0.5, 0.5], device=img.device).view(1, 1, 1, 2, 1)
        K_we_edge = torch.tensor([0.5, 0.5], device=img.device).view(1, 1, 1, 1, 2)

        ud_edges = F.relu(F.conv3d(-(2.0 * img - 1.0), K_ud_edge))
        num_ud_edges = F.avg_pool3d(ud_edges, (2, 3, 3), stride=2) * 18
        ns_edges = F.relu(F.conv3d(-(2.0 * img - 1.0), K_ns_edge))
        num_ns_edges = F.avg_pool3d(ns_edges, (3, 2, 3), stride=2) * 18
        we_edges = F.relu(F.conv3d(-(2.0 * img - 1.0), K_we_edge))
        num_we_edges = F.avg_pool3d(we_edges, (3, 3, 2), stride=2) * 18
        num_edges = num_ud_edges + num_ns_edges + num_we_edges

        masked_ud_edges = F.relu(F.conv3d(-(2.0 * masked_img - 1.0), K_ud_edge))
        num_masked_ud_edges = F.avg_pool3d(masked_ud_edges, (2, 3, 3), stride=2) * 18
        masked_ns_edges = F.relu(F.conv3d(-(2.0 * masked_img - 1.0), K_ns_edge))
        num_masked_ns_edges = F.avg_pool3d(masked_ns_edges, (3, 2, 3), stride=2) * 18
        masked_we_edges = F.relu(F.conv3d(-(2.0 * masked_img - 1.0), K_we_edge))
        num_masked_we_edges = F.avg_pool3d(masked_we_edges, (3, 3, 2), stride=2) * 18
        num_masked_edges = num_masked_ud_edges + num_masked_ns_edges + num_masked_we_edges

        # Count faces
        K_ud_face = torch.tensor([[0.25, 0.25], [0.25, 0.25]], device=img.device).view(1, 1, 1, 2, 2)
        K_ns_face = torch.tensor([[0.25, 0.25], [0.25, 0.25]], device=img.device).view(1, 1, 2, 1, 2)
        K_we_face = torch.tensor([[0.25, 0.25], [0.25, 0.25]], device=img.device).view(1, 1, 2, 2, 1)
        
        ud_faces = F.relu(F.conv3d(-(2.0 * img - 1.0), K_ud_face) - 0.5) * 2
        num_ud_faces = F.avg_pool3d(ud_faces, (3, 2, 2), stride=2) * 12
        ns_faces = F.relu(F.conv3d(-(2.0 * img - 1.0), K_ns_face) - 0.5) * 2
        num_ns_faces = F.avg_pool3d(ns_faces, (2, 3, 2), stride=2) * 12
        we_faces = F.relu(F.conv3d(-(2.0 * img - 1.0), K_we_face) - 0.5) * 2
        num_we_faces = F.avg_pool3d(we_faces, (2, 2, 3), stride=2) * 12
        num_faces = num_ud_faces + num_ns_faces + num_we_faces
        
        masked_ud_faces = F.relu(F.conv3d(-(2.0 * masked_img - 1.0), K_ud_face) - 0.5) * 2
        num_masked_ud_faces = F.avg_pool3d(masked_ud_faces, (3, 2, 2), stride=2) * 12
        masked_ns_faces = F.relu(F.conv3d(-(2.0 * masked_img - 1.0), K_ns_face) - 0.5) * 2
        num_masked_ns_faces = F.avg_pool3d(masked_ns_faces, (2, 3, 2), stride=2) * 12
        masked_we_faces = F.relu(F.conv3d(-(2.0 * masked_img - 1.0), K_we_face) - 0.5) * 2
        num_masked_we_faces = F.avg_pool3d(masked_we_faces, (2, 2, 3), stride=2) * 12
        num_masked_faces = num_masked_ud_faces + num_masked_ns_faces + num_masked_we_faces

        # Count octants
        K_octants = torch.tensor([[[0.125, 0.125], [0.125, 0.125]], [[0.125, 0.125], [0.125, 0.125]]], device=img.device).view(1, 1, 2, 2, 2)

        octants = F.relu(F.conv3d(-(2.0 * img - 1.0), K_octants) - 0.75) * 4
        num_octants = F.avg_pool3d(octants, (2, 2, 2), stride=2) * 8

        masked_octants = F.relu(F.conv3d(-(2.0 * masked_img - 1.0), K_octants) - 0.75) * 4
        num_masked_octants = F.avg_pool3d(masked_octants, (2, 2, 2), stride=2) * 8

        # Combined number of vertices, edges, faces and octants to calculate the euler characteristic
        euler_characteristic = num_vertices - num_edges + num_faces - num_octants
        masked_euler_characteristic = num_masked_vertices - num_masked_edges + num_masked_faces - num_masked_octants

        # If the Euler characteristic is unchanged after switching a point from 1 to 0 this indicates that the point is simple
        euler_change = F.hardtanh(torch.abs(masked_euler_characteristic - euler_characteristic), min_val=0, max_val=1)
        is_simple = 1 - euler_change
        is_simple = (is_simple.detach() > 0.5).float() - is_simple.detach() + is_simple

        return is_simple


    def _prepare_output(self, img):
        """
        Function that removes the padding and dimensions added by _prepare_input function.
        """

        img = img[:, :, 1:-1, 1:-1, 1:-1]

        if self.expanded_dims:
            img = torch.squeeze(img, dim=2)
        
        return img


    def _prepare_output(self, img):
        """
        Function that removes the padding and dimensions added by _prepare_input function.
        """

        img = img[:, :, 1:-1, 1:-1, 1:-1]

        if self.expanded_dims:
            img = torch.squeeze(img, dim=2)
        
        return img

class soft_cldice(nn.Module):
    def __init__(self, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.smooth = smooth
        # Skeletonize does not take in one-hot encodings
        self.skeletonize = Skeletonize(num_iter=5, probabilistic=True, beta=0.33)
        # We do not need prob here
        self.skeletonize_true = Skeletonize(num_iter=5, probabilistic=False)
        self.sigmoid = True
        self.fg_channel = 1

    def forward(self, y_true, y_pred):
        if self.sigmoid:
            y_pred = torch.nn.functional.softmax(y_pred,dim=1)
        if self.fg_channel:
            # Extract as slice
            y_pred = y_pred[:, self.fg_channel:(self.fg_channel+1), ...]
            #y_true = y_true[:, self.fg_channel:(self.fg_channel+1), ...]
        skel_pred = self.skeletonize(y_pred)
        skel_true = self.skeletonize_true(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice