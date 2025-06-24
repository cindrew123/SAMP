import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
from ..util import util as util
import cv2
from pytorch_msssim import ssim
no_ot = 0
style_Innovation = 0  # 1:OPEN the trans_style in cut,0 no.
IGNORE_LABEL = 255
import torch.nn as nn

Reinforcement_Learning = "TRUE"  # "TRUE","FALSE"


class SAMPModel(BaseModel):


    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.style_proj = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1),
            nn.Tanh()
        ).cuda()

        self.num_classes = 6


        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                          opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            # Ensure requires_grad is set correctly for network parameters
            for param in self.netG.parameters():
                param.requires_grad = True
            for param in self.netF.parameters():
                param.requires_grad = True
            for param in self.netD.parameters():
                param.requires_grad = True

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.attention_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).cuda()

    def output_loss(self):
        return self.loss_G_GAN, self.f_d_loss, self.loss_G, self.loss_D



    def data_dependent_initialize(self, data1, data2):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data1, data2)
        self.real_A.requires_grad_(True)
        self.real_B.requires_grad_(True)
        self.forward()  # compute fake images: G(A)
        print("inital again")
        print("isTrain is ", self.opt.isTrain)
        if self.opt.isTrain:
            self.compute_D_loss().backward(retain_graph=True)  # calculate gradients for D
            self.compute_G_loss().backward(retain_graph=True)  # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
                                                    betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def print_gradients(self, model):

        print("Gradients for each layer:")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Layer: {name} | requires_grad: {param.requires_grad}")
                if param.grad.dim() == 4:
                    print(f"Layer: {name} | Gradient: {param.grad[0, 0, :, :]}")
                elif param.grad.dim() == 3:
                    print(f"Layer: {name} | Gradient (sampled): {param.grad[0, :, :]}")
                elif param.grad.dim() == 2:
                    print(f"Layer: {name} | Gradient (sampled): {param.grad[:, :]}")
                else:
                    print(f"Layer: {name} | Gradient (full): {param.grad}")
            else:
                print(f"Layer: {name} | Gradient: None")

    def optimize_parameters(self, f_d_loss):
        self.f_d_loss = f_d_loss

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward(retain_graph=True)
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()

        if Reinforcement_Learning == "TRUE":
            pred_fake = self.netD(self.fake_B)
            reward = torch.mean(pred_fake).item()  # Calculate the average output of the discriminator as the reward
            adjusted_loss_G = self.loss_G * (1 + reward)
            adjusted_loss_G.backward()
        else:
            self.loss_G.backward(retain_graph=True)

        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()
        # self.print_gradients(self.netD)

    def set_input(self, real_A, real_B, PTS_labels=None):
        """Set the input images directly.
        Parameters:
            real_A (tensor): source domain image
            real_B (tensor): target domain ima
        """
        self.real_A = real_A.to(self.device).requires_grad_(True)
        self.real_B = real_B.to(self.device).requires_grad_(True)
        # Ensure the input dimensions are correct (4D tensor: [batch_size, channels, height, width])
        if len(self.real_A.shape) == 3:
            self.real_A = self.real_A.unsqueeze(0)
        if len(self.real_B.shape) == 3:
            self.real_B = self.real_B.unsqueeze(0)
        if PTS_labels is not None:
            self.PTS_labels = PTS_labels

    def get_generated_image(self, style_Innovation):
        """Return the generated fake images."""
        if style_Innovation == 1:
            # return self.fake_B,self.fake_B_tran,self.fake_B2A ,self.fake_B2A_tran,self.optimized_fake
            return self.fake_B, self.fake_B_tran, self.memory_out
        else:
            return self.fake_B

    def get_style_layers(self):
        if style_Innovation == 1:
            return self.content_layers_output, self.style_layers_output
        else:
            return self.fake_B, self.fake_B

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B),
                              dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        self.Contrastive_loss = self.calculate_Contrastive_loss(self.real_A.detach(), self.real_B.detach(),
                                                                self.fake_B.detach())

        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        # self.loss_G = self.loss_G_GAN + loss_NCE_both
        edge_mask = self.generate_edge_mask_batch(self.real_A)  # shape [B, 1, H, W]
        self.loss_structure_l1 = self.structure_preserving_l1_loss(fake, self.real_A, edge_mask)
        lambda_struct = 0.1
        self.loss_G = self.loss_G_GAN + loss_NCE_both + lambda_struct * self.loss_structure_l1

        return self.loss_G

    def generate_edge_mask_batch(self, images, threshold=100):
        masks = []
        for i in range(images.size(0)):
            img = images[i].detach().cpu()
            img_np = img.numpy()
            img_np = np.transpose(img_np, (1, 2, 0))  # [H, W, 3]
            img_np = (img_np * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, threshold1=threshold, threshold2=threshold * 2)
            edge_mask = (edges > 0).astype(np.float32)
            edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(images.device)  # [1, H, W]
            masks.append(edge_mask)
        return torch.stack(masks, dim=0)  # [B, 1, H, W]

    def structure_preserving_l1_loss(self, styled, content, edge_mask):
        if edge_mask.shape[1] == 1:
            edge_mask = edge_mask.repeat(1, 3, 1, 1)
        l1 = torch.abs(styled - content)
        masked_l1 = l1 * edge_mask
        return masked_l1.mean()

    def structure_preserving_ssim_loss(self, styled, content, edge_mask):
        B = styled.shape[0]
        total_loss = 0.0
        for i in range(B):
            s = styled[i].unsqueeze(0)
            c = content[i].unsqueeze(0)
            m = edge_mask[i].unsqueeze(0)
            s_masked = s * m
            c_masked = c * m
            score = ssim(s_masked, c_masked, data_range=1.0, size_average=True)
            total_loss += 1.0 - score
        return total_loss / B

    def calculate_Contrastive_loss(self, real_A, real_B, fake_B):
        # Get feature representation
        feat_real_A = self.netG(real_A, self.nce_layers, encode_only=True)
        feat_real_B = self.netG(real_B, self.nce_layers, encode_only=True)
        feat_fake_B = self.netG(fake_B, self.nce_layers, encode_only=True)
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_fake_B = [torch.flip(fg, [3]) for fg in feat_fake_B]
            feat_real_A = [torch.flip(fa, [3]) for fa in feat_real_A]
            feat_real_B = [torch.flip(fb, [3]) for fb in feat_real_B]
        feat_fake_B_pool, sample_ids = self.netF(feat_fake_B, self.opt.num_patches, None)
        feat_fake_real_A_pool, _ = self.netF(feat_real_A, self.opt.num_patches, sample_ids)
        feat_real_B_pool, _ = self.netF(feat_real_B, self.opt.num_patches, sample_ids)

        # Calculate positive sample contrast loss
        pos_loss = 0.0
        for f_tgt, f_pos in zip(feat_fake_B_pool, feat_real_B_pool):
            pos_loss += torch.nn.functional.mse_loss(f_tgt, f_pos)
        # Calculate negative sample contrast loss
        neg_loss = 0.0
        for f_tgt, f_src in zip(feat_fake_B_pool, feat_fake_real_A_pool):
            neg_loss += torch.nn.functional.mse_loss(f_tgt, f_src)
        margin = 0.1
        total_loss = torch.relu(pos_loss - neg_loss + margin)
        return total_loss

    def compute_semantic_consistency_loss(self, PTS, PTS_labels):
        """
        Calculate the semantic consistency loss using the generated image PTS and its labels.
        """
        if PTS.dtype != torch.float32:
            PTS = PTS.float()
        if PTS_labels.dtype != torch.long:
            PTS_labels = PTS_labels.long()
        if PTS.dim() != 4:
            return torch.tensor(0.0, device=PTS.device, requires_grad=True)
        else:
            if PTS_labels.dim() == 2:
                PTS_labels = PTS_labels.unsqueeze(0)
            criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL).cuda()
            loss = criterion(PTS, PTS_labels)
            return loss

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def apply_asch(self, styled_input, style_A, style_B, alpha=0.5):
        """
        Step C: Attention-Modulated Style-Content Harmonization（ASCH）
        - 在 SGAS 基础上进一步用注意力调和内容-风格权重
        """
        B, C, H, W = styled_input.shape
        if style_A.dim() == 3:
            style_A = style_A.unsqueeze(0)
        if style_B.dim() == 3:
            style_B = style_B.unsqueeze(0)

        style_A = torch.nn.functional.interpolate(style_A, size=(H, W), mode='bilinear', align_corners=False)
        style_B = torch.nn.functional.interpolate(style_B, size=(H, W), mode='bilinear', align_corners=False)

        attention_map = self.attention_net(styled_input)
        attention_map = attention_map.repeat(1, C, 1, 1)

        style_weight_A = 1 - attention_map
        style_weight_B = alpha * attention_map

        blended_style = style_weight_A * style_A + style_weight_B * styled_input
        cut_visuals = (1 - alpha) * style_A + alpha * styled_input
        cut_visuals = (1 - alpha) * blended_style + alpha * cut_visuals

        cut_visuals = self.match_histogram_std(style_B, cut_visuals)

        return cut_visuals

    def Style_Gradient(self, fake_B, style_A, style_B, PTS_labels=None, alpha=0.5):
        """
        联合调度：先SGAS后ASCH
        """
        # Step 1: 局部语义引导风格替换
        sgas_result = self.apply_sgas(fake_B, style_A, PTS_labels, alpha=alpha)

        # Step 2: 全图注意力风格调和
        cut_visuals = self.apply_asch(sgas_result, style_A, style_B, alpha=alpha)

        return cut_visuals

    def apply_sgas(self, fake_B, style_A, PTS_labels, alpha=0.5, use_structure_refine=True):
        """
        SGAS: 语义引导局部风格迁移 + 可选结构保持融合（支持多层特征融合）
        """
        if PTS_labels is None:
            return fake_B

        B, C, H, W = fake_B.shape

        if style_A.dim() == 3:
            style_A = style_A.unsqueeze(0)
        style_A = torch.nn.functional.interpolate(style_A, size=(H, W), mode='bilinear', align_corners=False)

        # === 多层风格特征提取与融合 ===
        selected_layers = [0, 3, 6]  # 可调整
        with torch.no_grad():
            feats = self.netG(style_A, selected_layers, encode_only=True)
            upsampled_feats = [torch.nn.functional.interpolate(f, size=(H, W), mode='bilinear', align_corners=False) for
                               f in feats]
            concat_feat = torch.cat(upsampled_feats, dim=1)  # [B, C_total, H, W]

        # === 投影为 style_map ===
        style_map = self.style_proj(concat_feat)  # shape: [B, 3, H, W]

        # === 按语义区域进行融合 ===
        fused_result = torch.zeros_like(fake_B)
        for cls in range(self.num_classes):
            mask = (PTS_labels == cls).float().unsqueeze(1).to(fake_B.device)  # shape: [B,1,H,W]
            mask = mask.repeat(1, 3, 1, 1)
            fused = alpha * style_map + (1 - alpha) * fake_B
            fused_result += fused * mask

        # === 可选结构保持 ===
        if use_structure_refine:
            fused_result = self.structure_guided_style_fusion(content=fake_B, styled=fused_result)

        return fused_result

    def structure_guided_style_fusion(self, content, styled, threshold=100, blend_strength=0.3):
        if content.dim() == 3:
            content = content.unsqueeze(0)
        if styled.dim() == 3:
            styled = styled.unsqueeze(0)
        assert content.shape == styled.shape, "content and styled must have the same shape"
        B, C, H, W = content.shape
        fused_batch = []

        for i in range(B):
            c_img = content[i].detach().cpu()
            if c_img.dim() != 3 or c_img.shape[0] != 3:
                raise ValueError(f"[Batch {i}] Expected shape [3,H,W], but got {c_img.shape}")

            c_img_np = np.transpose(c_img.numpy(), (1, 2, 0))  # [H, W, 3]
            c_img_np = (c_img_np * 255).astype(np.uint8)
            gray = cv2.cvtColor(c_img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, threshold1=threshold, threshold2=threshold * 2)
            edge_mask = (edges > 0).astype(np.float32)  # [H, W]
            edge_mask = torch.tensor(edge_mask).unsqueeze(0).repeat(3, 1, 1).to(content.device)  # [3, H, W]

            c_tensor = content[i]
            s_tensor = styled[i]
            fusion = edge_mask * c_tensor + (1 - edge_mask) * s_tensor
            final = blend_strength * fusion + (1 - blend_strength) * s_tensor
            fused_batch.append(final.unsqueeze(0))  # shape: [1, 3, H, W]

        return torch.cat(fused_batch, dim=0)  # shape: [B, 3, H, W]

    def match_histogram_simple(self, style_images, cut_visuals):
        style_max, style_min = style_images.max(), style_images.min()
        cut_max, cut_min = cut_visuals.max(), cut_visuals.min()
        normalized_cut_visuals = (cut_visuals - cut_min) / (cut_max - cut_min + 1e-8)
        matched_visuals = normalized_cut_visuals * (style_max - style_min) + style_min
        matched_visuals = torch.clamp(matched_visuals, min=style_min.item(), max=style_max.item())

        return matched_visuals

    def match_histogram_std(self, source, target):  # img target-->same as img source
        source_mean, source_std = torch.mean(source), torch.std(source)
        target_mean, target_std = torch.mean(target), torch.std(target)
        source_max, source_min = source.max(), source.min()
        matched = (target - target_mean) / (target_std + 1e-8) * (source_std + 1e-8) + source_mean
        matched = torch.clamp(matched, min=source_min.item(), max=source_max.item())
        return matched

    def match_histogram_max(self, source, target):  # img target-->same as img source
        source_mean, source_std = torch.mean(source), torch.std(source)
        target_mean, target_std = torch.mean(target), torch.std(target)

        matched = (target - target_mean) / (target_std + 1e-8) * (source_std + 1e-8) + source_mean

        source_max, source_min = source.max(), source.min()
        target_max, target_min = target.max(), target.min()

        target_range = target_max - target_min
        source_range = source_max - source_min

        matched = (matched - target_min) / (target_range + 1e-8) * (source_range + 1e-8) + source_min
        matched = torch.clamp(matched, min=source_min.item(), max=source_max.item())
        return matched



