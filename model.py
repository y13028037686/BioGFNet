
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)


class GrowthSupervisor(nn.Module):
    def __init__(self, in_ch=3, base_ch=16, out_ch=3):
        super().__init__()
        self.enc1 = ConvBlock3D(in_ch, base_ch)
        self.enc2 = ConvBlock3D(base_ch, base_ch*2)
        self.enc3 = ConvBlock3D(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool3d(2)
        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, 2, 2)
        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, 2, 2)
        self.dec2 = ConvBlock3D(base_ch*4, base_ch*2)
        self.dec1 = ConvBlock3D(base_ch*2, base_ch)
        self.final = nn.Conv3d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], 1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], 1))
        return torch.sigmoid(self.final(d1))




class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)  
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock3D(out_c)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, stride=stride),
                nn.InstanceNorm3d(out_c)
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return self.se(out)

class SimpleUNetEncoder(nn.Module):
    def __init__(self, in_c, base=16):
        super().__init__()
        self.enc1 = ResidualBlock(in_c, base)
        self.enc2 = ResidualBlock(base, base * 2, stride=2)
        self.enc3 = ResidualBlock(base * 2, base * 4, stride=2)
        self.enc4 = ResidualBlock(base * 4, base * 8, stride=2)
        self.bottleneck = ResidualBlock(base * 8, base * 16, stride=2)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        return e1, e2, e3, e4, b

class SimpleUNetDecoder(nn.Module):
    def __init__(self, out_c, base=16):
        super().__init__()
        self.up4 = nn.ConvTranspose3d(base * 16, base * 8, 2, stride=2)
        self.conv4 = nn.Conv3d(base * 16, base * 8, 3, padding=1)
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.conv3 = nn.Conv3d(base * 8, base * 4, 3, padding=1)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.conv2 = nn.Conv3d(base * 4, base * 2, 3, padding=1)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.conv1 = nn.Conv3d(base * 2, base, 3, padding=1)
        self.final = nn.Conv3d(base, out_c, 1)
        self.se4, self.se3, self.se2, self.se1 = (
            SEBlock3D(base * 8), SEBlock3D(base * 4),
            SEBlock3D(base * 2), SEBlock3D(base)
        )
    def forward(self, e1, e2, e3, e4, b):
        d4 = self.up4(b)
        d4 = torch.cat([self.se4(e4), d4], dim=1)
        d4 = F.relu(self.conv4(d4))
        d3 = self.up3(d4)
        d3 = torch.cat([self.se3(e3), d3], dim=1)
        d3 = F.relu(self.conv3(d3))
        d2 = self.up2(d3)
        d2 = torch.cat([self.se2(e2), d2], dim=1)
        d2 = F.relu(self.conv2(d2))
        d1 = self.up1(d2)
        d1 = torch.cat([self.se1(e1), d1], dim=1)
        d1 = F.relu(self.conv1(d1))
        return self.final(d1)

class HybridSegNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.encoder = SimpleUNetEncoder(in_channels)
        self.decoder = SimpleUNetDecoder(out_channels)
    def forward(self, x):
        e1, e2, e3, e4, b = self.encoder(x)
        return self.decoder(e1, e2, e3, e4, b)



base_model_path = os.path.join( "best_model.pth")
assert os.path.exists(base_model_path), "❌ "

stage1_model = HybridSegNet().to(device)
stage1_model.load_state_dict(torch.load(base_model_path)["model"])
stage1_model.eval()  



growth_supervisor = GrowthSupervisor().to(device)
growth_supervisor.load_state_dict(torch.load("label2growth_best240*240*160.pth"))
growth_supervisor.eval()



class KANResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_c)

        self.kan_branch = nn.Sequential(
            nn.Conv3d(out_c, out_c, 1),
            nn.SiLU(),
            nn.Conv3d(out_c, out_c, 1)
        )

        self.se = SEBlock3D(out_c, reduction)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, stride=stride),
                nn.InstanceNorm3d(out_c)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))  

        out = out + 0.3 * self.kan_branch(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.se(out + identity)
        return self.relu(out)


class KANAttention3D(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.se = SEBlock3D(ch, reduction)

        self.spatial = nn.Sequential(
            nn.Conv3d(ch, ch // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch // reduction, 1, kernel_size=7, padding=3),
            nn.Sigmoid()        
        )

        self.kan = nn.Sequential(
            nn.Conv3d(ch, ch, 1),
            nn.SiLU(),
            nn.Conv3d(ch, ch, 1)
        )

    def forward(self, x):
        x_se = self.se(x)

        spatial_mask = self.spatial(x_se)
        x_spatial = x_se * spatial_mask

        x_kan = self.kan(x_spatial)

        return x + 0.5 * x_kan




class MultiAttentionDecoder(nn.Module):
    def __init__(self, out_c, base=16):
        super().__init__()
        self.up4 = nn.ConvTranspose3d(base * 16, base * 8, 2, stride=2)
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)

        self.att4 = KANAttention3D(base * 8)
        self.att3 = KANAttention3D(base * 4)
        self.att2 = KANAttention3D(base * 2)
        self.att1 = KANAttention3D(base)

        self.conv4 = nn.Conv3d(base * 16, base * 8, 3, padding=1)
        self.conv3 = nn.Conv3d(base * 8, base * 4, 3, padding=1)
        self.conv2 = nn.Conv3d(base * 4, base * 2, 3, padding=1)
        self.conv1 = nn.Conv3d(base * 2, base, 3, padding=1)
        self.final = nn.Conv3d(base, out_c, 1)

    def forward(self, e1, e2, e3, e4, b):
        d4 = self.up4(b)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = F.relu(self.conv4(d4))
        d4 = self.att4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = F.relu(self.conv3(d3))
        d3 = self.att3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = F.relu(self.conv2(d2))
        d2 = self.att2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = F.relu(self.conv1(d1))
        d1 = self.att1(d1)

        return self.final(d1)



class HybridSegNet_Fusion_Plus(nn.Module):
    def __init__(self, in_channels=4, stage1_channels=3, growth_channels=3, out_channels=3, base=32):
        super().__init__()
        self.encoder = nn.ModuleDict({
            'enc1': KANResidualBlock(in_channels + stage1_channels, base),
            'enc2': KANResidualBlock(base, base * 2, stride=2),
            'enc3': KANResidualBlock(base * 2, base * 4, stride=2),
            'enc4': KANResidualBlock(base * 4, base * 8, stride=2),
            'bottleneck': KANResidualBlock(base * 8, base * 16, stride=2)
        })

        self.decoder = MultiAttentionDecoder(out_c=out_channels, base=base)

        self.cross_attn_e1 = CrossAttentionBlock(dim_q=base, dim_kv=growth_channels)
        self.cross_attn_e2 = CrossAttentionBlock(dim_q=base * 2, dim_kv=growth_channels)
        self.cross_attn_e3 = CrossAttentionBlock(dim_q=base * 4, dim_kv=growth_channels)
        self.alpha1 = nn.Parameter(torch.tensor(0.6))
        self.alpha2 = nn.Parameter(torch.tensor(0.6))
        self.alpha3 = nn.Parameter(torch.tensor(0.6))

    def forward(self, image, rough_pred, growth_tensor):
        x = torch.cat([image, rough_pred], dim=1)
        e1 = self.encoder['enc1'](x)
        e1 = e1 + self.alpha1 * cross_attention_fusion(e1, growth_tensor, self.cross_attn_e1)
        e2 = self.encoder['enc2'](e1)
        e2 = e2 + self.alpha2 * cross_attention_fusion(e2, growth_tensor, self.cross_attn_e2)
        e3 = self.encoder['enc3'](e2)
        e3 = e3 + self.alpha3 * cross_attention_fusion(e3, growth_tensor, self.cross_attn_e3)
        e4 = self.encoder['enc4'](e3)
        b = self.encoder['bottleneck'](e4)
        return self.decoder(e1, e2, e3, e4, b)



class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim_q // num_heads) ** -0.5
        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.v_proj = nn.Linear(dim_kv, dim_q)
        self.out_proj = nn.Linear(dim_q, dim_q)

    def forward(self, q, kv):
        B, N, C = q.shape
        _, M, _ = kv.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(kv).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        v = self.v_proj(kv).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, N, C//heads]
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)


def cross_attention_fusion(images, growth_pred, cross_attn_module, target_size=(16, 16, 16)):
    B, C, D, H, W = images.shape
    images_ds = F.interpolate(images, size=target_size, mode='trilinear', align_corners=False)
    growth_ds = F.interpolate(growth_pred, size=target_size, mode='trilinear', align_corners=False)
    q = images_ds.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
    kv = growth_ds.permute(0, 2, 3, 4, 1).reshape(B, -1, growth_pred.shape[1])
    attn_out = cross_attn_module(q, kv)
    attn_out = attn_out.reshape(B, *target_size, C).permute(0, 4, 1, 2, 3)
    attn_out = F.interpolate(attn_out, size=(D, H, W), mode='trilinear', align_corners=False)
    fused = images + attn_out
    return fused




import torch.nn as nn
from monai.losses import DiceLoss

def weighted_hybrid_loss(pred, target):

    dice_loss = DiceLoss(sigmoid=True, reduction='mean')
    ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    
    loss_wt = dice_loss(pred[:,0:1], target[:,0:1]) + 0.3 * ce_loss(pred[:,0:1], target[:,0:1])
    loss_tc = dice_loss(pred[:,1:2], target[:,1:2]) + 0.3 * ce_loss(pred[:,1:2], target[:,1:2])
    loss_et = dice_loss(pred[:,2:3], target[:,2:3]) + 0.3 * ce_loss(pred[:,2:3], target[:,2:3])
    
    return 1.0 * loss_wt + 1.2 * loss_tc + 2.0 * loss_et  

loss_fn = weighted_hybrid_loss


