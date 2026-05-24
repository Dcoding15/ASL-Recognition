import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# CNN LSTM
# --------------------------

class CNNLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=1):
        super(CNNLSTM, self).__init__()
        
        # CNN feature extractor (simple CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Compute flattened CNN feature size
        self.flatten_dim = 128 * 8 * 8  # for input 64x64 after 3 poolings
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.flatten_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Fully connected classification layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        
        # Flatten sequence dimension into batch
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)  # Reshape to (batch, seq, features)
        
        lstm_out, _ = self.lstm(x)
        
        # Use output of the last frame
        out = self.fc(lstm_out[:, -1, :])
        return out


# --------------------------
# CNN Bi-LSTM
# --------------------------

class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=1):
        super(CNNBiLSTM, self).__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.flatten_dim = 128 * 8 * 8  # 8192

        # 👉 BiLSTM (bidirectional=True)
        self.lstm = nn.LSTM(
            input_size=self.flatten_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 👉 Hidden size doubled (forward + backward)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(x)

        # 👉 Use last time step output (includes both directions)
        out = self.fc(lstm_out[:, -1, :])
        return out


# --------------------------
# CNN Bi-LSTM with Attention
# --------------------------

class AttentionCNNBiLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=1):
        super(AttentionCNNBiLSTM, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        
        self.flatten_dim = 128 * 8 * 8  # 8192 features

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=self.flatten_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Attention Layer
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()

        # CNN over each frame
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (B, S, 2H)

        # Attention Mechanism
        attn_weights = self.attention(lstm_out)         # (B, S, 1)
        attn_weights = F.softmax(attn_weights, dim=1)   # Normalized weights

        # Weighted sum of LSTM outputs
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # (B, 2H)

        # Classification
        out = self.fc(context_vector)
        return out


# --------------------------
# 3D Convolutional Neural Network (CNN)
# --------------------------

class CNN3D(nn.Module):
    def __init__(self, num_classes):
        super(CNN3D, self).__init__()

        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),  # <-- Updated

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
        )

        self.fc1 = nn.Linear(128 * 1 * 8 * 8, 256)  
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)

        x = self.cnn3d(x)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --------------------------
# Vision Image Transformer (ViT)
# --------------------------

class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=8, num_classes=10, dim=256, depth=6, heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()

        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_size = patch_size

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_dim, dim)

        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=depth
        )

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def patchify(self, images):
        """
        images: (B, C, H, W)
        returns: (B, num_patches, patch_dim)
        """
        B, C, H, W = images.shape
        p = self.patch_size
        patches = images.unfold(2, p, p).unfold(3, p, p)  # (B, C, H//p, W//p, p, p)
        patches = patches.contiguous().view(B, -1, C * p * p)
        return patches

    def forward(self, x):
        """
        x: (B, T, C, H, W) sequence of frames
        returns: (B, num_classes) averaged over frames
        """
        B, T, C, H, W = x.shape

        # Flatten batch and time
        x = x.view(B * T, C, H, W)          # (B*T, C, H, W)
        x = self.patchify(x)                # (B*T, num_patches, patch_dim)
        x = self.patch_embedding(x)         # (B*T, num_patches, dim)

        # Class token
        cls_tokens = self.cls_token.expand(B * T, 1, -1)  # (B*T, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)             # (B*T, num_patches+1, dim)

        # Add positional embedding
        x += self.pos_embedding

        # Transformer
        x = self.transformer(x)            # (B*T, num_patches+1, dim)

        # Classification head on cls token
        x = self.mlp_head(x[:, 0])         # (B*T, num_classes)

        # Reshape back to (B, T, num_classes) and average over time
        x = x.view(B, T, -1).mean(dim=1)   # (B, num_classes)

        return x

# --------------------------
# Temporal Convolutional Network (TCN)
# --------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, num_classes=10):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation_size, padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
    
    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)        # (B, C, T)
        y = self.network(x)          # (B, C_out, T)
        y = y.mean(dim=2)            # global temporal average pooling
        y = self.fc(y)               # (B, num_classes)
        return y

# --------------------------
# Frame feature encoder
# --------------------------
class FrameFeatureEncoder(nn.Module):
    """
    Reduces flattened frame features to manageable dimension
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        B, T, D = x.shape
        x = self.fc(x.view(B*T, D))
        x = self.relu(x)
        return x.view(B, T, -1)  # (B, T, hidden_dim)

# --------------------------
# Two-Stream Network
# --------------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.conv(x)         # (B, feature_dim, 1, 1)
        x = x.view(x.size(0), -1)  # (B, feature_dim)
        return x

class TwoStreamStatic(nn.Module):
    def __init__(self, feature_dim=256, num_classes=10):
        super().__init__()
        self.spatial_cnn = SimpleCNN(in_channels=3, feature_dim=feature_dim)
        self.temporal_cnn = SimpleCNN(in_channels=1, feature_dim=feature_dim)  # edge / gradient stream
        self.fc = nn.Linear(feature_dim * 2, num_classes)

    def forward(self, x):
        """
        x: (B, 3, H, W)
        """
        # --- Spatial stream ---
        spatial_feat = self.spatial_cnn(x)  # (B, feature_dim)

        # --- Temporal / edge stream ---
        # Compute simple edge map using gradients
        gray = 0.2989 * x[:,0] + 0.5870 * x[:,1] + 0.1140 * x[:,2]  # grayscale
        gray = gray.unsqueeze(1)  # (B, 1, H, W)
        gx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        gy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        edge_map = F.pad(gx, (0,1,0,0)) + F.pad(gy, (0,0,0,1))  # simple edge
        temporal_feat = self.temporal_cnn(edge_map)

        # --- Fusion ---
        feat = torch.cat([spatial_feat, temporal_feat], dim=1)
        out = self.fc(feat)
        return out
