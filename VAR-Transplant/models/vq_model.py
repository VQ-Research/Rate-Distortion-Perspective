import os
import sys
import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from models.vanilla_quantizer import VanillaVectorQuantizer
from models.ema_quantizer import EMAVectorQuantizer
from models.online_quantizer import OnlineVectorQuantizer
from models.wasserstein_quantizer import WassersteinVectorQuantizer
from models.mmd_quantizer import MMDVectorQuantizer
from models.encoder_decoder import Encoder, Decoder, Normalize
from utils.util import Pack
from safetensors.torch import load_file
from models.lpips import LPIPS

class VQModel(nn.Module):
    def __init__(self, args):
        super(VQModel, self).__init__()
        self.args = args
        ddconfig = dict(
            dropout=0, ch=160, z_channels=32,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   
            using_sa=True, using_mid_sa=True,                          
        )
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.quant_conv = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.post_quant_conv = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)

        if args.VQ == "vanilla_vq":
            if args.chunks == 1:
                self.quantizer1 = VanillaVectorQuantizer(args)
            elif args.chunks == 2:
                self.quantizer1 = VanillaVectorQuantizer(args)
                self.quantizer2 = VanillaVectorQuantizer(args)
            elif args.chunks == 4:
                self.quantizer1 = VanillaVectorQuantizer(args)
                self.quantizer2 = VanillaVectorQuantizer(args)
                self.quantizer3 = VanillaVectorQuantizer(args)
                self.quantizer4 = VanillaVectorQuantizer(args)
            elif args.chunks == 8:
                self.quantizer1 = VanillaVectorQuantizer(args)
                self.quantizer2 = VanillaVectorQuantizer(args)
                self.quantizer3 = VanillaVectorQuantizer(args)
                self.quantizer4 = VanillaVectorQuantizer(args)
                self.quantizer5 = VanillaVectorQuantizer(args)
                self.quantizer6 = VanillaVectorQuantizer(args)
                self.quantizer7 = VanillaVectorQuantizer(args)
                self.quantizer8 = VanillaVectorQuantizer(args)
        elif args.VQ == "ema_vq":
            if args.chunks == 1:
                self.quantizer1 = EMAVectorQuantizer(args)
            elif args.chunks == 2:
                self.quantizer1 = EMAVectorQuantizer(args)
                self.quantizer2 = EMAVectorQuantizer(args)
            elif args.chunks == 4:
                self.quantizer1 = EMAVectorQuantizer(args)
                self.quantizer2 = EMAVectorQuantizer(args)
                self.quantizer3 = EMAVectorQuantizer(args)
                self.quantizer4 = EMAVectorQuantizer(args)
            elif args.chunks == 8:
                self.quantizer1 = EMAVectorQuantizer(args)
                self.quantizer2 = EMAVectorQuantizer(args)
                self.quantizer3 = EMAVectorQuantizer(args)
                self.quantizer4 = EMAVectorQuantizer(args)
                self.quantizer5 = EMAVectorQuantizer(args)
                self.quantizer6 = EMAVectorQuantizer(args)
                self.quantizer7 = EMAVectorQuantizer(args)
                self.quantizer8 = EMAVectorQuantizer(args)
        elif args.VQ == "online_vq":
            if args.chunks == 1:
                self.quantizer1 = OnlineVectorQuantizer(args)
            elif args.chunks == 2:
                self.quantizer1 = OnlineVectorQuantizer(args)
                self.quantizer2 = OnlineVectorQuantizer(args)
            elif args.chunks == 4:
                self.quantizer1 = OnlineVectorQuantizer(args)
                self.quantizer2 = OnlineVectorQuantizer(args)
                self.quantizer3 = OnlineVectorQuantizer(args)
                self.quantizer4 = OnlineVectorQuantizer(args)
            elif args.chunks == 8:
                self.quantizer1 = OnlineVectorQuantizer(args)
                self.quantizer2 = OnlineVectorQuantizer(args)
                self.quantizer3 = OnlineVectorQuantizer(args)
                self.quantizer4 = OnlineVectorQuantizer(args)
                self.quantizer5 = OnlineVectorQuantizer(args)
                self.quantizer6 = OnlineVectorQuantizer(args)
                self.quantizer7 = OnlineVectorQuantizer(args)
                self.quantizer8 = OnlineVectorQuantizer(args)
        elif args.VQ == "wasserstein_vq":
            if args.chunks == 1:
                self.quantizer1 = WassersteinVectorQuantizer(args)
            elif args.chunks == 2:
                self.quantizer1 = WassersteinVectorQuantizer(args)
                self.quantizer2 = WassersteinVectorQuantizer(args)
            elif args.chunks == 4:
                self.quantizer1 = WassersteinVectorQuantizer(args)
                self.quantizer2 = WassersteinVectorQuantizer(args)
                self.quantizer3 = WassersteinVectorQuantizer(args)
                self.quantizer4 = WassersteinVectorQuantizer(args)
            elif args.chunks == 8:
                self.quantizer1 = WassersteinVectorQuantizer(args)
                self.quantizer2 = WassersteinVectorQuantizer(args)
                self.quantizer3 = WassersteinVectorQuantizer(args)
                self.quantizer4 = WassersteinVectorQuantizer(args)
                self.quantizer5 = WassersteinVectorQuantizer(args)
                self.quantizer6 = WassersteinVectorQuantizer(args)
                self.quantizer7 = WassersteinVectorQuantizer(args)
                self.quantizer8 = WassersteinVectorQuantizer(args)
        elif args.VQ == "mmd_vq":
            if args.chunks == 1:
                self.quantizer1 = MMDVectorQuantizer(args)
            elif args.chunks == 2:
                self.quantizer1 = MMDVectorQuantizer(args)
                self.quantizer2 = MMDVectorQuantizer(args)
            elif args.chunks == 4:
                self.quantizer1 = MMDVectorQuantizer(args)
                self.quantizer2 = MMDVectorQuantizer(args)
                self.quantizer3 = MMDVectorQuantizer(args)
                self.quantizer4 = MMDVectorQuantizer(args)
            elif args.chunks == 8:
                self.quantizer1 = MMDVectorQuantizer(args)
                self.quantizer2 = MMDVectorQuantizer(args)
                self.quantizer3 = MMDVectorQuantizer(args)
                self.quantizer4 = MMDVectorQuantizer(args)
                self.quantizer5 = MMDVectorQuantizer(args)
                self.quantizer6 = MMDVectorQuantizer(args)
                self.quantizer7 = MMDVectorQuantizer(args)
                self.quantizer8 = MMDVectorQuantizer(args)

        self.projector_in = nn.Sequential(
                nn.Conv2d(32, 1024, kernel_size=3, padding=1),
                Normalize(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                Normalize(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 32, kernel_size=3, padding=1),
            )
        self.projector_out = nn.Sequential(
                nn.Conv2d(32, 1024, kernel_size=3, padding=1),
                Normalize(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                Normalize(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 32, kernel_size=3, padding=1),
            )

        if args.stage == "transplant":
            self.perceptual_loss = LPIPS().eval()
            pretrain_dict = torch.load(args.pretrained_tokenizer, map_location='cpu', weights_only=False)
            encoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('encoder.')}
            decoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('decoder.')}
            quant_conv_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quant_conv.')}
            post_quant_conv_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('post_quant_conv.')}

            encoder_dict = {k.replace('encoder.', '', 1): v for k, v in encoder_dict.items()}
            decoder_dict = {k.replace('decoder.', '', 1): v for k, v in decoder_dict.items()}
            quant_conv_dict = {k.replace('quant_conv.', '', 1): v for k, v in quant_conv_dict.items()}
            post_quant_conv_dict = {k.replace('post_quant_conv.', '', 1): v for k, v in post_quant_conv_dict.items()}

            self.encoder.load_state_dict(encoder_dict, strict=True)
            self.decoder.load_state_dict(decoder_dict, strict=True)
            self.quant_conv.load_state_dict(quant_conv_dict, strict=True)
            self.post_quant_conv.load_state_dict(post_quant_conv_dict, strict=True)

            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.quant_conv.parameters():
                param.requires_grad = False
            for param in self.post_quant_conv.parameters():
                param.requires_grad = False
            for param in self.projector_in.parameters():
                param.requires_grad = True
            for param in self.projector_out.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = False

            if args.chunks == 1:
                for param in self.quantizer1.parameters():
                    param.requires_grad = True
            elif args.chunks == 2:
                for param in self.quantizer1.parameters():
                    param.requires_grad = True 
                for param in self.quantizer2.parameters():
                    param.requires_grad = True
            elif args.chunks == 4:
                for param in self.quantizer1.parameters():
                    param.requires_grad = True 
                for param in self.quantizer2.parameters():
                    param.requires_grad = True
                for param in self.quantizer3.parameters():
                    param.requires_grad = True 
                for param in self.quantizer4.parameters():
                    param.requires_grad = True
            elif args.chunks == 8:
                for param in self.quantizer1.parameters():
                    param.requires_grad = True 
                for param in self.quantizer2.parameters():
                    param.requires_grad = True
                for param in self.quantizer3.parameters():
                    param.requires_grad = True 
                for param in self.quantizer4.parameters():
                    param.requires_grad = True
                for param in self.quantizer5.parameters():
                    param.requires_grad = True 
                for param in self.quantizer6.parameters():
                    param.requires_grad = True
                for param in self.quantizer7.parameters():
                    param.requires_grad = True 
                for param in self.quantizer8.parameters():
                    param.requires_grad = True
            
            self.encoder.eval()
            self.decoder.eval()
            self.quant_conv.eval()
            self.post_quant_conv.eval()

        if args.stage == "refinement":
            checkpoint_dir = os.path.join(os.path.join(args.init_checkpoint_dir, "Transplant"), args.dataset_name)
            checkpoint_name = args.checkpoint_name
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

            pretrain_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model']
            encoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('encoder.')}
            decoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('decoder.')}
            quant_conv_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quant_conv.')}
            post_quant_conv_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('post_quant_conv.')}
            projector_in_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('projector_in.')}
            projector_out_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('projector_out.')}
            if args.chunks == 1:
                quantizer1_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer1.')}
            elif args.chunks == 2:
                quantizer1_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer1.')}
                quantizer2_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer2.')}
            elif args.chunks == 4:
                quantizer1_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer1.')}
                quantizer2_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer2.')}
                quantizer3_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer3.')}
                quantizer4_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer4.')}
            elif args.chunks == 8:
                quantizer1_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer1.')}
                quantizer2_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer2.')}
                quantizer3_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer3.')}
                quantizer4_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer4.')}
                quantizer5_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer5.')}
                quantizer6_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer6.')}
                quantizer7_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer7.')}
                quantizer8_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer8.')}


            encoder_dict = {k.replace('encoder.', '', 1): v for k, v in encoder_dict.items()}
            decoder_dict = {k.replace('decoder.', '', 1): v for k, v in decoder_dict.items()}
            quant_conv_dict = {k.replace('quant_conv.', '', 1): v for k, v in quant_conv_dict.items()}
            post_quant_conv_dict = {k.replace('post_quant_conv.', '', 1): v for k, v in post_quant_conv_dict.items()}
            projector_in_dict = {k.replace('projector_in.', '', 1): v for k, v in projector_in_dict.items()}
            projector_out_dict = {k.replace('projector_out.', '', 1): v for k, v in projector_out_dict.items()}
            if args.chunks == 1:
                quantizer1_dict = {k.replace('quantizer1.', '', 1): v for k, v in quantizer1_dict.items()}
            elif args.chunks == 2:
                quantizer1_dict = {k.replace('quantizer1.', '', 1): v for k, v in quantizer1_dict.items()}
                quantizer2_dict = {k.replace('quantizer2.', '', 1): v for k, v in quantizer2_dict.items()}
            elif args.chunks == 4:
                quantizer1_dict = {k.replace('quantizer1.', '', 1): v for k, v in quantizer1_dict.items()}
                quantizer2_dict = {k.replace('quantizer2.', '', 1): v for k, v in quantizer2_dict.items()}
                quantizer3_dict = {k.replace('quantizer3.', '', 1): v for k, v in quantizer3_dict.items()}
                quantizer4_dict = {k.replace('quantizer4.', '', 1): v for k, v in quantizer4_dict.items()}
            elif args.chunks == 8:
                quantizer1_dict = {k.replace('quantizer1.', '', 1): v for k, v in quantizer1_dict.items()}
                quantizer2_dict = {k.replace('quantizer2.', '', 1): v for k, v in quantizer2_dict.items()}
                quantizer3_dict = {k.replace('quantizer3.', '', 1): v for k, v in quantizer3_dict.items()}
                quantizer4_dict = {k.replace('quantizer4.', '', 1): v for k, v in quantizer4_dict.items()}
                quantizer5_dict = {k.replace('quantizer5.', '', 1): v for k, v in quantizer5_dict.items()}
                quantizer6_dict = {k.replace('quantizer6.', '', 1): v for k, v in quantizer6_dict.items()}
                quantizer7_dict = {k.replace('quantizer7.', '', 1): v for k, v in quantizer7_dict.items()}
                quantizer8_dict = {k.replace('quantizer8.', '', 1): v for k, v in quantizer8_dict.items()}

            self.encoder.load_state_dict(encoder_dict, strict=True)
            self.decoder.load_state_dict(decoder_dict, strict=True)
            self.quant_conv.load_state_dict(quant_conv_dict, strict=True)
            self.post_quant_conv.load_state_dict(post_quant_conv_dict, strict=True)
            self.projector_in.load_state_dict(projector_in_dict, strict=True)
            self.projector_out.load_state_dict(projector_out_dict, strict=True)
            if args.chunks == 1:
                self.quantizer1.load_state_dict(quantizer1_dict, strict=True)
            elif args.chunks == 2:
                self.quantizer1.load_state_dict(quantizer1_dict, strict=True)
                self.quantizer2.load_state_dict(quantizer2_dict, strict=True)
            elif args.chunks == 4:
                self.quantizer1.load_state_dict(quantizer1_dict, strict=True)
                self.quantizer2.load_state_dict(quantizer2_dict, strict=True)
                self.quantizer3.load_state_dict(quantizer3_dict, strict=True)
                self.quantizer4.load_state_dict(quantizer4_dict, strict=True)
            elif args.chunks == 8:
                self.quantizer1.load_state_dict(quantizer1_dict, strict=True)
                self.quantizer2.load_state_dict(quantizer2_dict, strict=True)
                self.quantizer3.load_state_dict(quantizer3_dict, strict=True)
                self.quantizer4.load_state_dict(quantizer4_dict, strict=True)
                self.quantizer5.load_state_dict(quantizer5_dict, strict=True)
                self.quantizer6.load_state_dict(quantizer6_dict, strict=True)
                self.quantizer7.load_state_dict(quantizer7_dict, strict=True)
                self.quantizer8.load_state_dict(quantizer8_dict, strict=True)            

            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.quant_conv.parameters():
                param.requires_grad = False          
            for param in self.projector_in.parameters():
                param.requires_grad = False
            for param in self.projector_out.parameters():
                param.requires_grad = True
            for param in self.post_quant_conv.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True
            if args.chunks == 1:
                for param in self.quantizer1.parameters():
                    param.requires_grad = False
            elif args.chunks == 2:
                for param in self.quantizer1.parameters():
                    param.requires_grad = False
                for param in self.quantizer2.parameters():
                    param.requires_grad = False
            elif args.chunks == 4:
                for param in self.quantizer1.parameters():
                    param.requires_grad = False
                for param in self.quantizer2.parameters():
                    param.requires_grad = False
                for param in self.quantizer3.parameters():
                    param.requires_grad = False
                for param in self.quantizer4.parameters():
                    param.requires_grad = False
            elif args.chunks == 8:
                for param in self.quantizer1.parameters():
                    param.requires_grad = False
                for param in self.quantizer2.parameters():
                    param.requires_grad = False
                for param in self.quantizer3.parameters():
                    param.requires_grad = False
                for param in self.quantizer4.parameters():
                    param.requires_grad = False
                for param in self.quantizer5.parameters():
                    param.requires_grad = False
                for param in self.quantizer6.parameters():
                    param.requires_grad = False
                for param in self.quantizer7.parameters():
                    param.requires_grad = False
                for param in self.quantizer8.parameters():
                    param.requires_grad = False 

            self.encoder.eval()
            self.quant_conv.eval()
            self.projector_in.eval()
            if args.chunks == 1:
                self.quantizer1.eval()
            elif args.chunks == 2:
                self.quantizer1.eval()
                self.quantizer2.eval()
            elif args.chunks == 4:
                self.quantizer1.eval()
                self.quantizer2.eval()
                self.quantizer3.eval()
                self.quantizer4.eval()
            elif args.chunks == 8:
                self.quantizer1.eval()
                self.quantizer2.eval()
                self.quantizer3.eval()
                self.quantizer4.eval()
                self.quantizer5.eval()
                self.quantizer6.eval()
                self.quantizer7.eval()
                self.quantizer8.eval()    

    def transplant(self, x):
        assert self.args.stage == "transplant"
        with torch.no_grad():
            ze = self.encoder(x)
            z_obj = self.quant_conv(ze)

        z_p = z_obj + self.projector_in(z_obj)
        if self.args.chunks == 1:
            z_q, vq_loss, utilization, perplexity = self.quantizer1(z_p)
        elif self.args.chunks == 2:
            z_p_1, z_p_2 = torch.chunk(z_p, 2, dim=1)
            z_q_1, vq_loss_1, utilization_1, perplexity_1 = self.quantizer1(z_p_1)
            z_q_2, vq_loss_2, utilization_2, perplexity_2 = self.quantizer2(z_p_2)
            z_q = torch.cat((z_q_1, z_q_2), dim=1)
            vq_loss = (vq_loss_1 + vq_loss_2) * 0.5
            utilization = (utilization_1 + utilization_2) * 0.5
            perplexity = (perplexity_1 + perplexity_2) * 0.5
        elif self.args.chunks == 4:
            z_p_1, z_p_2, z_p_3, z_p_4 = torch.chunk(z_p, 4, dim=1)
            z_q_1, vq_loss_1, utilization_1, perplexity_1 = self.quantizer1(z_p_1)
            z_q_2, vq_loss_2, utilization_2, perplexity_2 = self.quantizer2(z_p_2)
            z_q_3, vq_loss_3, utilization_3, perplexity_3 = self.quantizer3(z_p_3)
            z_q_4, vq_loss_4, utilization_4, perplexity_4 = self.quantizer4(z_p_4)
            z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)
            vq_loss = (vq_loss_1 + vq_loss_2 + vq_loss_3 + vq_loss_4) * 0.25
            utilization = (utilization_1 + utilization_2 + utilization_3 + utilization_4) * 0.25
            perplexity = (perplexity_1 + perplexity_2 + perplexity_3 + perplexity_4) * 0.25
        elif self.args.chunks == 8:
            z_p_1, z_p_2, z_p_3, z_p_4, z_p_5, z_p_6, z_p_7, z_p_8 = torch.chunk(z_p, 8, dim=1)
            z_q_1, vq_loss_1, utilization_1, perplexity_1 = self.quantizer1(z_p_1)
            z_q_2, vq_loss_2, utilization_2, perplexity_2 = self.quantizer2(z_p_2)
            z_q_3, vq_loss_3, utilization_3, perplexity_3 = self.quantizer3(z_p_3)
            z_q_4, vq_loss_4, utilization_4, perplexity_4 = self.quantizer4(z_p_4)
            z_q_5, vq_loss_5, utilization_5, perplexity_5 = self.quantizer5(z_p_5)
            z_q_6, vq_loss_6, utilization_6, perplexity_6 = self.quantizer6(z_p_6)
            z_q_7, vq_loss_7, utilization_7, perplexity_7 = self.quantizer7(z_p_7)
            z_q_8, vq_loss_8, utilization_8, perplexity_8 = self.quantizer8(z_p_8)
            z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4, z_q_5, z_q_6, z_q_7, z_q_8), dim=1)
            vq_loss = (vq_loss_1 + vq_loss_2 + vq_loss_3 + vq_loss_4 + vq_loss_5 + vq_loss_6 + vq_loss_7 + vq_loss_8) * 0.125
            utilization = (utilization_1 + utilization_2 + utilization_3 + utilization_4 + utilization_5 + utilization_6 + utilization_7 + utilization_8) * 0.125
            perplexity = (perplexity_1 + perplexity_2 + perplexity_3 + perplexity_4 + perplexity_5 + perplexity_6 + perplexity_7 + perplexity_8) * 0.125
        
        z_q = z_q + self.projector_out(z_q)
        loss = F.mse_loss(z_q, z_obj.detach())
        quant_error = F.mse_loss(z_q.detach(), z_obj.detach())
        z_q = self.post_quant_conv(z_q)
        x_rec = self.decoder(z_q)

        p_loss = self.perceptual_loss(x.contiguous(), x_rec.contiguous())
        p_loss = torch.mean(p_loss)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        transplant_loss = 5.0 * rec_loss + p_loss + loss + vq_loss
        return  transplant_loss, rec_loss, p_loss, quant_error, utilization, perplexity

    def refinement(self, x):
        assert self.args.stage == "refinement"
        with torch.no_grad():
            ze = self.encoder(x)
            z_obj = self.quant_conv(ze)
            z_p = z_obj + self.projector_in(z_obj)

            if self.args.chunks == 1:
                z_q, _ = self.quantizer1.collect_eval_info(z_p)
            elif self.args.chunks == 2:
                z_p_1, z_p_2 = torch.chunk(z_p, 2, dim=1)
                z_q_1, _ = self.quantizer1.collect_eval_info(z_p_1)
                z_q_2, _ = self.quantizer2.collect_eval_info(z_p_2)
                z_q = torch.cat((z_q_1, z_q_2), dim=1)
            elif self.args.chunks == 4:
                z_p_1, z_p_2, z_p_3, z_p_4 = torch.chunk(z_p, 4, dim=1)
                z_q_1, _ = self.quantizer1.collect_eval_info(z_p_1)
                z_q_2, _ = self.quantizer2.collect_eval_info(z_p_2)
                z_q_3, _ = self.quantizer3.collect_eval_info(z_p_3)
                z_q_4, _ = self.quantizer4.collect_eval_info(z_p_4)
                z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)
            elif self.args.chunks == 8:
                z_p_1, z_p_2, z_p_3, z_p_4, z_p_5, z_p_6, z_p_7, z_p_8 = torch.chunk(z_p, 8, dim=1)
                z_q_1, _ = self.quantizer1.collect_eval_info(z_p_1)
                z_q_2, _ = self.quantizer2.collect_eval_info(z_p_2)
                z_q_3, _ = self.quantizer3.collect_eval_info(z_p_3)
                z_q_4, _ = self.quantizer4.collect_eval_info(z_p_4)
                z_q_5, _ = self.quantizer5.collect_eval_info(z_p_5)
                z_q_6, _ = self.quantizer6.collect_eval_info(z_p_6)
                z_q_7, _ = self.quantizer7.collect_eval_info(z_p_7)
                z_q_8, _ = self.quantizer8.collect_eval_info(z_p_8)
                z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4, z_q_5, z_q_6, z_q_7, z_q_8), dim=1)
        
        z_q = z_q + self.projector_out(z_q)
        z_q = self.post_quant_conv(z_q)
        x_rec = self.decoder(z_q)
        return x_rec

    def collect_eval_info_transplant(self, x):
        ze = self.encoder(x)
        z_obj = self.quant_conv(ze)

        z_p = z_obj + self.projector_in(z_obj)
        if self.args.chunks == 1:
            z_q, histogram_1 = self.quantizer1.collect_eval_info(z_p)
        elif self.args.chunks == 2:
            z_p_1, z_p_2 = torch.chunk(z_p, 2, dim=1)
            z_q_1, histogram_1 = self.quantizer1.collect_eval_info(z_p_1)
            z_q_2, histogram_2 = self.quantizer2.collect_eval_info(z_p_2)
            z_q = torch.cat((z_q_1, z_q_2), dim=1)
        elif self.args.chunks == 4:
            z_p_1, z_p_2, z_p_3, z_p_4 = torch.chunk(z_p, 4, dim=1)
            z_q_1, histogram_1 = self.quantizer1.collect_eval_info(z_p_1)
            z_q_2, histogram_2 = self.quantizer2.collect_eval_info(z_p_2)
            z_q_3, histogram_3 = self.quantizer3.collect_eval_info(z_p_3)
            z_q_4, histogram_4 = self.quantizer4.collect_eval_info(z_p_4)
            z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)
        elif self.args.chunks == 8:
            z_p_1, z_p_2, z_p_3, z_p_4, z_p_5, z_p_6, z_p_7, z_p_8 = torch.chunk(z_p, 8, dim=1)
            z_q_1, histogram_1 = self.quantizer1.collect_eval_info(z_p_1)
            z_q_2, histogram_2 = self.quantizer2.collect_eval_info(z_p_2)
            z_q_3, histogram_3 = self.quantizer3.collect_eval_info(z_p_3)
            z_q_4, histogram_4 = self.quantizer4.collect_eval_info(z_p_4)
            z_q_5, histogram_5 = self.quantizer5.collect_eval_info(z_p_5)
            z_q_6, histogram_6 = self.quantizer6.collect_eval_info(z_p_6)
            z_q_7, histogram_7 = self.quantizer7.collect_eval_info(z_p_7)
            z_q_8, histogram_8 = self.quantizer8.collect_eval_info(z_p_8)
            z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4, z_q_5, z_q_6, z_q_7, z_q_8), dim=1)
        z_q = z_q + self.projector_out(z_q)

        quant_error = F.mse_loss(z_q.detach(), z_obj.detach())
        z_q = self.post_quant_conv(z_q)
        x_rec = self.decoder(z_q).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        if self.args.chunks == 1:
            return x_rec, rec_loss, quant_error, histogram_1
        elif self.args.chunks == 2:
            return x_rec, rec_loss, quant_error, histogram_1, histogram_2
        elif self.args.chunks == 4:
            return x_rec, rec_loss, quant_error, histogram_1, histogram_2, histogram_3, histogram_4
        elif self.args.chunks == 8:
            return x_rec, rec_loss, quant_error, histogram_1, histogram_2, histogram_3, histogram_4, histogram_5, histogram_6, histogram_7, histogram_8

    def collect_eval_info_refinement(self, x):
        ze = self.encoder(x)
        z_obj = self.quant_conv(ze)

        z_p = z_obj + self.projector_in(z_obj)
        if self.args.chunks == 1:
            z_q, _ = self.quantizer1.collect_eval_info(z_p)
        elif self.args.chunks == 2:
            z_p_1, z_p_2 = torch.chunk(z_p, 2, dim=1)
            z_q_1, _ = self.quantizer1.collect_eval_info(z_p_1)
            z_q_2, _ = self.quantizer2.collect_eval_info(z_p_2)
            z_q = torch.cat((z_q_1, z_q_2), dim=1)
        elif self.args.chunks == 4:
            z_p_1, z_p_2, z_p_3, z_p_4 = torch.chunk(z_p, 4, dim=1)
            z_q_1, _ = self.quantizer1.collect_eval_info(z_p_1)
            z_q_2, _ = self.quantizer2.collect_eval_info(z_p_2)
            z_q_3, _ = self.quantizer3.collect_eval_info(z_p_3)
            z_q_4, _ = self.quantizer4.collect_eval_info(z_p_4)
            z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)
        elif self.args.chunks == 8:
            z_p_1, z_p_2, z_p_3, z_p_4, z_p_5, z_p_6, z_p_7, z_p_8 = torch.chunk(z_p, 8, dim=1)
            z_q_1, _ = self.quantizer1.collect_eval_info(z_p_1)
            z_q_2, _ = self.quantizer2.collect_eval_info(z_p_2)
            z_q_3, _ = self.quantizer3.collect_eval_info(z_p_3)
            z_q_4, _ = self.quantizer4.collect_eval_info(z_p_4)
            z_q_5, _ = self.quantizer5.collect_eval_info(z_p_5)
            z_q_6, _ = self.quantizer6.collect_eval_info(z_p_6)
            z_q_7, _ = self.quantizer7.collect_eval_info(z_p_7)
            z_q_8, _ = self.quantizer8.collect_eval_info(z_p_8)
            z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4, z_q_5, z_q_6, z_q_7, z_q_8), dim=1)
        z_q = z_q + self.projector_out(z_q)

        z_q = self.post_quant_conv(z_q)
        x_rec = self.decoder(z_q).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss
        
    def reconstruction(self, x):
        ze = self.encoder(x)
        z_obj = self.quant_conv(ze)

        z_p = z_obj + self.projector_in(z_obj)
        if self.args.chunks == 1:
            z_q = self.quantizer1.collect_reconstruction(z_p)
        elif self.args.chunks == 2:
            z_p_1, z_p_2 = torch.chunk(z_p, 2, dim=1)
            z_q_1 = self.quantizer1.collect_reconstruction(z_p_1)
            z_q_2 = self.quantizer2.collect_reconstruction(z_p_2)
            z_q = torch.cat((z_q_1, z_q_2), dim=1)
        elif self.args.chunks == 4:
            z_p_1, z_p_2, z_p_3, z_p_4 = torch.chunk(z_p, 4, dim=1)
            z_q_1 = self.quantizer1.collect_reconstruction(z_p_1)
            z_q_2 = self.quantizer2.collect_reconstruction(z_p_2)
            z_q_3 = self.quantizer3.collect_reconstruction(z_p_3)
            z_q_4 = self.quantizer4.collect_reconstruction(z_p_4)
            z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)
        elif self.args.chunks == 8:
            z_p_1, z_p_2, z_p_3, z_p_4, z_p_5, z_p_6, z_p_7, z_p_8 = torch.chunk(z_p, 8, dim=1)
            z_q_1 = self.quantizer1.collect_reconstruction(z_p_1)
            z_q_2 = self.quantizer2.collect_reconstruction(z_p_2)
            z_q_3 = self.quantizer3.collect_reconstruction(z_p_3)
            z_q_4 = self.quantizer4.collect_reconstruction(z_p_4)
            z_q_5 = self.quantizer5.collect_reconstruction(z_p_5)
            z_q_6 = self.quantizer6.collect_reconstruction(z_p_6)
            z_q_7 = self.quantizer7.collect_reconstruction(z_p_7)
            z_q_8 = self.quantizer8.collect_reconstruction(z_p_8)
            z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4, z_q_5, z_q_6, z_q_7, z_q_8), dim=1)
        z_q = z_q + self.projector_out(z_q)

        z_q = self.post_quant_conv(z_q)
        x_rec = self.decoder(z_q).clamp_(-1, 1)
        return x_rec

