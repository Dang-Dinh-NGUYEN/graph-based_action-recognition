import math
import random
import torch
import torch.nn.functional as F
import numpy as np

# --- Temporal Augmentation ---
class TemporalCrop:
    def __init__(self, padding_ratio=6):
        """
        Args:
            padding_ratio: determines padding length as T // padding_ratio
        """
        self.padding_ratio = padding_ratio

    @torch.no_grad()
    def __call__(self, data):
        """
        Args:
            data: Tensor or ndarray [N, C, T, V, M] or [C, T, V, M]
        Returns:
            data_cropped: same type (torch.Tensor or ndarray)
        """
        # --- Handle numpy input ---
        
        is_numpy = isinstance(data, np.ndarray)
        if is_numpy:
            data = torch.from_numpy(data).float()

        # --- Handle single-sample input ---
        if data.ndim == 4:
            data = data.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        N, C, T, V, M = data.shape
        device = data.device
        data = data.to(device)
        padding_len = max(1, T // self.padding_ratio)  # avoid zero pad length

        # --- Mirror padding ---
        start_pad = torch.flip(data[:, :, :padding_len, :, :], dims=[2])
        end_pad   = torch.flip(data[:, :, -padding_len:, :, :], dims=[2])
        data_padded = torch.cat([start_pad, data, end_pad], dim=2)  # [N, C, T+2*pad, V, M]

        # --- Random crop indices ---
        frame_start = torch.randint(0, 2 * padding_len + 1, (N,), device=device)
        idx = torch.arange(T, device=device).view(1, -1) + frame_start.view(-1, 1)  # [N, T]

        # --- Gather using broadcasted indices ---
        idx = idx.view(N, 1, T, 1, 1).expand(-1, C, -1, V, M)
        data_cropped = torch.gather(data_padded, dim=2, index=idx)

        # --- Restore shape ---
        if squeeze_back:
            data_cropped = data_cropped.squeeze(0)

        if is_numpy:
            data_cropped = data_cropped.cpu().numpy()

        return data_cropped

class RandomTemporalPermutation:
    def __init__(self, max_segments=5):
        """
        max_segments: the number of segments to split the sequence into before permuting.
                      Prevents completely chaotic frame shuffling and keeps local structure.
        """
        self.max_segments = max_segments

    def __call__(self, x):
        """
        x: Tensor of shape [C, T, V] (common format for skeleton data)
        """
        if not isinstance(x, torch.Tensor) or x.dim() != 5:
            raise ValueError("Expected tensor of shape [N, C, T, V, M]")

        N, C, T, V, M = x.shape

        # Decide number of segments (at least 2 if T is big enough)
        num_segments = min(self.max_segments, T)
        if num_segments < 2:
            return x  # not enough frames to permute

        # Split into segments
        segment_lengths = [T // num_segments] * num_segments
        for i in range(T % num_segments):
            segment_lengths[i] += 1  # handle uneven split

        # Compute segment boundaries
        segments = []
        start = 0
        for length in segment_lengths:
            segments.append(x[:, start:start + length, :, :])
            start += length

        # Permute segments
        permuted_indices = torch.randperm(len(segments))
        permuted = [segments[i] for i in permuted_indices]

        # Concatenate back
        x_permuted = torch.cat(permuted, dim=1)  # along time axis

        return x_permuted

class TemporalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            data = data.flip(dims=[2])  # in-place flip is fine
        return data
    
class TemporalGaussianBlur:
    def __init__(self, p=0.5, sigma_range=(0.1, 2.0), kernel_size=15):
        self.p = p
        self.sigma_range = sigma_range
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def _create_kernel(self, sigma):
        """Create 1D Gaussian kernel."""
        half = self.kernel_size // 2
        t = torch.arange(-half, half + 1, dtype=torch.float32)
        kernel = torch.exp(-t**2 / (2 * sigma**2))
        kernel /= kernel.sum()
        return kernel.view(1, 1, -1)  # shape: (1, 1, K)

    def __call__(self, x):
        """
        Args:
            x: Tensor of shape (N, C=3, T, V, M)
        Returns:
            Blurred tensor of same shape
        """
        if random.random() > self.p:
            return x

        sigma = random.uniform(*self.sigma_range)
        kernel = self._create_kernel(sigma).to(x.device)  # (1, 1, K)

        N, C, T, V, M = x.shape
        x = x.permute(0, 3, 4, 1, 2).reshape(-1, C, T)  # (N*V*M, C, T)
        x = F.pad(x, (self.padding, self.padding), mode='reflect')
        x = F.conv1d(x, kernel.expand(C, 1, -1), groups=C)
        x = x.view(N, V, M, C, T).permute(0, 3, 4, 1, 2)  # (N, C, T, V, M)

        return x
    
# --- Rotation ---
def get_rotation_matrices(angles_rad, axes, device='cpu'):
    """
    angles_rad: (N,) tensor of angles
    axes: list of 'x', 'y', 'z', length N
    returns: (N, 3, 3) rotation matrices
    """
    N = angles_rad.shape[0]
    R = torch.eye(3, device=device).unsqueeze(0).repeat(N,1,1)  # (N,3,3)

    c = torch.cos(angles_rad)
    s = torch.sin(angles_rad)

    for i, axis in enumerate(axes):
        if axis == 'x':
            R[i] = torch.tensor([[1,0,0],
                                 [0,c[i],-s[i]],
                                 [0,s[i],c[i]]], device=device)
        elif axis == 'y':
            R[i] = torch.tensor([[c[i],0,s[i]],
                                 [0,1,0],
                                 [-s[i],0,c[i]]], device=device)
        elif axis == 'z':
            R[i] = torch.tensor([[c[i],-s[i],0],
                                 [s[i],c[i],0],
                                 [0,0,1]], device=device)
        else:
            raise ValueError(f"Invalid axis: {axis}")
    return R  # (N,3,3)


class RandomFrameRotation:
    def __init__(self, angle_range=30, axes=('x', 'y', 'z'), max_frames=5):
        self.angle_range = angle_range
        self.axes = axes
        self.max_frames = max_frames

    def __call__(self, x):
        B, C, T, V, M = x.shape
        device = x.device
        if C != 3:
            raise ValueError("Expected C=3 for rotation")

        with torch.no_grad():
            # Step 1: select random frames
            num_frames_per_sample = torch.randint(1, min(self.max_frames, T)+1, (B,), device=device)
            frame_masks = torch.zeros((B, T), dtype=torch.bool, device=device)
            for i, nf in enumerate(num_frames_per_sample):
                frame_masks[i, torch.randperm(T, device=device)[:nf]] = True

            # Step 2: flatten selected frames
            batch_idx, frame_idx = torch.where(frame_masks)  # (N_selected,)
            N_selected = batch_idx.shape[0]

            if N_selected == 0:
                return x  # nothing to rotate

            # Step 3: generate random rotation matrices
            angles_rad = (torch.rand(N_selected, device=device) * 2 - 1) * math.radians(self.angle_range)
            axes_choice = [random.choice(self.axes) for _ in range(N_selected)]
            R = get_rotation_matrices(angles_rad, axes_choice, device=device)  # (N_selected,3,3)

            # Step 4: gather selected frames
            x_selected = x[batch_idx, :, frame_idx, :, :]  # (N_selected,3,V,M)
            x_selected_flat = x_selected.reshape(N_selected, 3, V*M)  # (N_selected,3, V*M)

            # Step 5: apply rotation
            x_rotated_flat = torch.bmm(R, x_selected_flat)  # (N_selected,3, V*M)
            x_rotated = x_rotated_flat.view(N_selected, 3, V, M)

            # Step 6: scatter back
            x = x.clone()
            x[batch_idx, :, frame_idx, :, :] = x_rotated

            return x

class RandomRotation:
    def __init__(self, main_angle_max=math.pi / 6, secondary_angle_max=math.pi / 180):
        self.main_angle_max = main_angle_max
        self.secondary_angle_max = secondary_angle_max
        self.axis_indices = {'X': 0, 'Y': 1, 'Z': 2}

    def __call__(self, data):
        """
        Args:
            data: Tensor of shape (N, C=3, T, V, M)
        Returns:
            Rotated tensor of same shape
        """
        assert data.shape[1] == 3, "Rotation requires 3D coordinates (C=3)"
        device = data.device
        with torch.no_grad():
            N, C, T, V, M = data.shape

            # Select main axis and angles
            axes = ['X', 'Y', 'Z']
            main_axis = random.choice(axes)
            angles = {}

            for ax in axes:
                if ax == main_axis:
                    angles[ax] = random.uniform(0, self.main_angle_max)
                else:
                    angles[ax] = random.uniform(0, self.secondary_angle_max)

            # Rotation matrices
            Rx = torch.tensor([
                [1, 0, 0],
                [0, math.cos(angles['X']), -math.sin(angles['X'])],
                [0, math.sin(angles['X']), math.cos(angles['X'])]
            ], device=device)

            Ry = torch.tensor([
                [math.cos(angles['Y']), 0, math.sin(angles['Y'])],
                [0, 1, 0],
                [-math.sin(angles['Y']), 0, math.cos(angles['Y'])]
            ], device=device)

            Rz = torch.tensor([
                [math.cos(angles['Z']), -math.sin(angles['Z']), 0],
                [math.sin(angles['Z']), math.cos(angles['Z']), 0],
                [0, 0, 1]
            ], device=device)

            # Combined rotation
            R = Rz @ Ry @ Rx  # Note: matrix multiplication order matters

            # Reshape and apply rotation
            data_reshaped = data.permute(0, 2, 3, 4, 1).reshape(-1, 3)  # (N*T*V*M, 3)
            rotated = (R @ data_reshaped.T).T  # Apply rotation
            rotated = rotated.reshape(N, T, V, M, 3).permute(0, 4, 1, 2, 3)  # back to (N, C, T, V, M)

        return rotated
    
class AllFrameRotation:
    def __init__(self, axes=("x", "y", "z"), angles=(0, math.pi/2, math.pi, 3*math.pi/2)):
        """
        Applies the same rotation matrix to all frames of a sample.

        Args:
            axes (tuple): Axes to rotate around.
            angles (tuple): Angles (in radians) to sample from.
        """
        self.rotations = self._generate_rotations(axes, angles)

    def __call__(self, x):
        """
        Args:
            x (Tensor): Shape [C=3, T, V, M] or [B, C=3, T, V, M]
        Returns:
            Tensor: Rotated tensor of the same shape.
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError("Expected input to be a torch.Tensor")

        if x.dim() == 5:
            B, C, T, V, M = x.shape
            if C != 3:
                raise ValueError("Expected C=3 for rotation")
            R = random.choice(self.rotations).to(x.device)  # [3, 3]
            x_flat = x.view(B, C, -1)  # [B, 3, T*V*M]
            x_rot = torch.bmm(R.expand(B, -1, -1), x_flat)  # [B, 3, T*V*M]
            return x_rot.view(B, C, T, V, M)

        elif x.dim() == 4:
            C, T, V, M = x.shape
            if C != 3:
                raise ValueError("Expected C=3 for rotation")
            R = random.choice(self.rotations).to(x.device)
            x_flat = x.view(C, -1)  # [3, T*V*M]
            x_rot = R @ x_flat
            return x_rot.view(C, T, V, M)

        else:
            raise ValueError("Expected tensor of shape [C, T, V, M] or [B, C, T, V, M]")

    @staticmethod
    def _generate_rotations(axes, angles):
        """Generates rotation matrices for all axis-angle pairs."""
        return [get_rotation_matrices(theta, axis) for axis in axes for theta in angles]

# --- Other spatial Augmentation ---

class RandomGaussianNoise:
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, x):
        return torch.randn_like(x) * self.std
    
class RandomScale:
    def __init__(self, scale_range=(0.8, 1.2)):
        """
        scale_range: tuple (min, max) for uniform scaling
        """
        self.min_scale, self.max_scale = scale_range

    def __call__(self, x):
        """
        x: tensor of shape [C, T, V] (e.g., [3, 300, 25])
        """
        if not isinstance(x, torch.Tensor) or x.dim() != 5:
            raise ValueError("Expected tensor of shape [B, C, T, V, M]")

        scale = torch.empty(1).uniform_(self.min_scale, self.max_scale).item()
        return x * scale

class Shear:
    def __init__(self, beta=0.5):
        """
        Args:
            beta: maximum absolute value of shear coefficients
        """
        self.beta = beta

    @torch.no_grad()
    def __call__(self, data):
        """
        Args:
            data: Tensor or ndarray [N, C=3, T, V, M] or [C=3, T, V, M]
        Returns:
            data_sheared: same type (torch.Tensor or ndarray)
        """
        # Convert numpy to torch (preserve device)
        is_numpy = isinstance(data, np.ndarray)
        if is_numpy:
            data = torch.from_numpy(data).float()

        # Add batch dimension if needed
        if data.ndim == 4:
            data = data.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        N, C, T, V, M = data.shape
        if C != 3:
            raise ValueError(f"Expected C=3 for shear, got C={C}")

        device = data.device
        data = data.to(device)

        # Random shear params
        s1 = (torch.rand(N, 3, device=device) * 2 - 1) * self.beta
        s2 = (torch.rand(N, 3, device=device) * 2 - 1) * self.beta

        # Build shear matrices [N, 3, 3]
        R = torch.zeros(N, 3, 3, device=device)
        R[:, 0, 0] = 1; R[:, 0, 1] = s1[:, 0]; R[:, 0, 2] = s2[:, 0]
        R[:, 1, 0] = s1[:, 1]; R[:, 1, 1] = 1; R[:, 1, 2] = s2[:, 1]
        R[:, 2, 0] = s1[:, 2]; R[:, 2, 1] = s2[:, 2]; R[:, 2, 2] = 1

        # Flatten and apply shear
        x_flat = data.view(N, C, -1)
        x_sheared = torch.bmm(R, x_flat)
        data_sheared = x_sheared.view(N, C, T, V, M)

        if squeeze_back:
            data_sheared = data_sheared.squeeze(0)

        # Convert back to numpy if needed
        if is_numpy:
            data_sheared = data_sheared.cpu().numpy()

        return data_sheared
    
class SpatialFlip:
    def __init__(self, left_joints=[4, 5, 6, 7, 21, 22], right_joints=[8, 9, 10, 11, 23, 24], p=0.5):
        """
        Args:
            left_joints: list of indices for left-side joints
            right_joints: list of corresponding right-side joints
            p: probability of applying the flip
        """
        assert len(left_joints) == len(right_joints), "Left and right joint lists must match"
        self.left_joints = left_joints
        self.right_joints = right_joints
        self.p = p

    def __call__(self, data):
        """
        Args:
            data (torch.Tensor): (N, C, T, V, M)
        Returns:
            Flipped or original tensor
        """
        if random.random() > self.p:
            return data  # no flip

        # Clone to avoid in-place ops
        data = data.clone()
        
        # Swap left/right joints
        data_flipped = data.clone()
        for l, r in zip(self.left_joints, self.right_joints):
            data_flipped[:, :, :, l, :] = data[:, :, :, r, :]
            data_flipped[:, :, :, r, :] = data[:, :, :, l, :]
        
        return data_flipped

class AxisMask:
    def __init__(self, p=0.5):
        self.p = p
        self.axis_names = ['X', 'Y', 'Z']

    def __call__(self, data):
        """
        Args:
            data: Tensor of shape (N, C=3, T, V, M)
        Returns:
            Augmented tensor of same shape with one axis masked (zeroed) with probability p
        """
        if random.random() < self.p:
            axis_to_mask = random.randint(0, 2)  # 0: X, 1: Y, 2: Z
            data = data.clone()  # avoid in-place ops
            data[:, axis_to_mask, :, :, :] = 0
        return data
    

def shear(data_numpy, r=0.5):
    s1_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]
    s2_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]

    R = np.array([[1,          s1_list[0], s2_list[0]],
                  [s1_list[1], 1,          s2_list[1]],
                  [s1_list[2], s2_list[2], 1        ]])

    R = R.transpose()
    data_numpy = np.dot(data_numpy.transpose([1, 2, 3, 0]), R)
    data_numpy = data_numpy.transpose(3, 0, 1, 2)
    return data_numpy


def temperal_crop(data_numpy, temperal_padding_ratio=6):
    C, T, V, M = data_numpy.shape
    padding_len = T // temperal_padding_ratio
    frame_start = np.random.randint(0, padding_len * 2 + 1)
    data_numpy = np.concatenate((data_numpy[:, :padding_len][:, ::-1],
                                 data_numpy,
                                 data_numpy[:, -padding_len:][:, ::-1]),
                                axis=1)
    data_numpy = data_numpy[:, frame_start:frame_start + T]
    return data_numpy