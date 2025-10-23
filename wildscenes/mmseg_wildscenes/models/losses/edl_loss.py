import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS

@MODELS.register_module()
class EDLSegLoss(nn.Module):
    """性能优化的证据深度学习分割损失
    
    完全适配 mmseg，只改损失计算方式
    """
    
    def __init__(self,
                 num_classes,
                 annealing_steps=10000,
                 kl_weight=0.2,
                 ignore_index=255,
                 loss_weight=1.0,
                 loss_name='loss_edl'):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_steps = annealing_steps
        self.kl_weight = kl_weight
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        
        # 训练步数计数器
        self.register_buffer('step_counter', torch.zeros(1, dtype=torch.long))
    
    def forward(self, alpha, target):
        """
        Args:
            alpha: Dirichlet 参数 (B, C, H, W)
            target: 标签 (B, H, W)
        Returns:
            loss_dict: 损失字典（mmseg 标准格式）
        """
        B, C, H, W = alpha.shape
        
        # 展平
        alpha = alpha.permute(0, 2, 3, 1).contiguous().view(-1, C)
        target = target.view(-1)
        
        # 过滤无效像素
        mask = target != self.ignore_index
        alpha = alpha[mask]
        target = target[mask]
        
        if alpha.size(0) == 0:
            return {self._loss_name: alpha.new_zeros(1) * 0.0}
        
        # 更新步数
        if self.training:
            self.step_counter += 1
        
        # 退火系数（从 0 到 1）
        annealing = torch.clamp(
            self.step_counter.float() / self.annealing_steps, 
            0.0, 1.0
        ).item()
        
        # One-hot 标签
        y_onehot = F.one_hot(target, self.num_classes).float()
        
        # 主损失：Type-II Maximum Likelihood
        S = alpha.sum(dim=1, keepdim=True)
        main_loss = self.typeII_ml_loss(alpha, S, y_onehot)
        
        # KL 正则化
        kl_loss = self.compute_kl_reg(alpha, y_onehot)
        
        # 总损失（带退火）
        total = (main_loss + annealing * self.kl_weight * kl_loss) * self.loss_weight
        
        # 返回字典（mmseg 标准）
        return {self._loss_name: total}
    
    def typeII_ml_loss(self, alpha, S, y):
        """Type-II Maximum Likelihood 损失（论文标准实现）"""
        psi_S = torch.digamma(S)
        psi_alpha = torch.digamma(alpha)
        loss = -(y * (psi_alpha - psi_S)).sum(dim=1).mean()
        return loss
    
    def compute_kl_reg(self, alpha, y):
        """KL 散度正则化（论文标准实现）"""
        alpha_tilde = y + (1.0 - y) * alpha
        S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
        K = float(self.num_classes)
        
        term1 = torch.lgamma(S_tilde) - torch.lgamma(
            torch.tensor(K, dtype=alpha.dtype, device=alpha.device)
        )
        term2 = -torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
        term3 = (
            (alpha_tilde - 1.0) * 
            (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))
        ).sum(dim=1, keepdim=True)
        
        kl = (term1 + term2 + term3).mean()
        return kl
    
    @property
    def loss_name(self):
        """返回损失名称（mmseg 标准接口）"""
        return self._loss_name