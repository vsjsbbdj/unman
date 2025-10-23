import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.psp_head import PPM
from mmseg.registry import MODELS
from mmseg.models.utils import resize

@MODELS.register_module()
class EDLUPerHead(BaseDecodeHead):
    """证据深度学习 UPerNet 解码头（完整版）
    
    自动适配：
    - 多层输入（主解码头）：完整 PSP + FPN
    - 单层输入（辅助头）：只用 PSP
    """
    
    def __init__(self,
                 pool_scales=(1, 2, 3, 6),
                 evidence_func='exp',
                 **kwargs):
        # ===== 关键修改：自动判断单层/多层 =====
        if 'in_index' in kwargs:
            if isinstance(kwargs['in_index'], (list, tuple)) and len(kwargs['in_index']) > 1:
                # 多层输入 -> multiple_select
                kwargs.setdefault('input_transform', 'multiple_select')
            else:
                # 单层输入 -> resize_concat（或不设置）
                kwargs.setdefault('input_transform', None)
        # ==========================================
        
        super().__init__(**kwargs)
        
        self.evidence_func = evidence_func
        
        # 判断是否多尺度
        self.is_multi_scale = isinstance(self.in_index, (list, tuple)) and len(self.in_index) > 1
        
        # PSP 模块
        psp_in_channels = self.in_channels[-1] if self.is_multi_scale else self.in_channels
        self.psp_head = PPM(
            pool_scales,
            psp_in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        
        # PSP 输出融合
        self.bottleneck = ConvModule(
            psp_in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        # FPN 模块（仅多尺度时需要）
        if self.is_multi_scale:
            self.lateral_convs = nn.ModuleList()
            self.fpn_convs = nn.ModuleList()
            
            for i in range(len(self.in_channels) - 1):
                lateral = ConvModule(
                    self.in_channels[i],
                    self.channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
                
                fpn_conv = ConvModule(
                    self.channels,
                    self.channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
                
                self.lateral_convs.append(lateral)
                self.fpn_convs.append(fpn_conv)
            
            # FPN 多尺度融合
            self.fpn_bottleneck = ConvModule(
                len(self.in_channels) * self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        
        # 最终分类层
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
    
    def psp_forward(self, x):
        """PSP 分支（单层特征）"""
        psp_outs = [x]
        psp_outs.extend(self.psp_head(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output
    
    def forward(self, inputs):
        """前向传播
        
        Returns:
            alpha: Dirichlet 参数 (B, C, H, W)
        """
        inputs = self._transform_inputs(inputs)
        
        # ===== 单层输入：只用 PSP =====
        if not self.is_multi_scale:
            output = self.psp_forward(inputs)
        
        # ===== 多层输入：PSP + FPN =====
        else:
            # PSP 分支
            psp_out = self.psp_forward(inputs[-1])
            
            # FPN 横向连接
            laterals = [
                lateral_conv(inputs[i])
                for i, lateral_conv in enumerate(self.lateral_convs)
            ]
            laterals.append(psp_out)
            
            # FPN 自顶向下
            used_backbone_levels = len(laterals)
            for i in range(used_backbone_levels - 1, 0, -1):
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + resize(
                    laterals[i],
                    size=prev_shape,
                    mode='bilinear',
                    align_corners=self.align_corners)
            
            # FPN 输出精炼
            fpn_outs = [
                self.fpn_convs[i](laterals[i])
                for i in range(used_backbone_levels - 1)
            ]
            fpn_outs.append(laterals[-1])
            
            # 多尺度拼接
            target_size = fpn_outs[0].shape[2:]
            fpn_outs = [
                resize(
                    out,
                    size=target_size,
                    mode='bilinear',
                    align_corners=self.align_corners
                ) for out in fpn_outs
            ]
            
            fpn_out = torch.cat(fpn_outs, dim=1)
            output = self.fpn_bottleneck(fpn_out)
        
        # logits -> evidence -> α
        logits = self.conv_seg(output)
        evidence = self.get_evidence(logits)
        alpha = evidence + 1.0
        
        return alpha
    
    def get_evidence(self, logits):
        """logits -> 非负证据"""
        if self.evidence_func == 'exp':
            return torch.exp(torch.clamp(logits, -10, 10))
        elif self.evidence_func == 'softplus':
            return F.softplus(logits)
        else:
            return F.relu(logits)
    
    def loss_by_feat(self, seg_logits, batch_data_samples):
        """损失计算"""
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = self.loss_decode(seg_logits, seg_label)
        
        # 精度
        S = seg_logits.sum(dim=1, keepdim=True)
        prob = seg_logits / S
        pred = prob.argmax(dim=1)
        
        mask = seg_label != self.ignore_index
        if mask.sum() > 0:
            correct = (pred[mask] == seg_label[mask]).sum()
            total = mask.sum()
            acc_seg = correct.float() / total.float() * 100.0
        else:
            acc_seg = seg_logits.new_zeros(1)
        
        loss['acc_seg'] = acc_seg
        return loss
    
    def predict_by_feat(self, seg_logits, batch_img_metas):
        """预测"""
        S = seg_logits.sum(dim=1, keepdim=True)
        prob = seg_logits / S
        
        seg_probs = resize(
            input=prob,
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners)
        
        return seg_probs