import numpy as np
import scipy as sp
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from util.spectro_img import compute_visuals
from util.util import kbdwin
from .base_model import BaseModel
from . import networks
from .mdct import MDCT4, IMDCT4
#from dct.dct_native import DCT_2N_native, IDCT_2N_native
import torchaudio.functional as aF

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_match_loss, use_time_loss, use_mr_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, use_match_loss, use_time_loss, use_time_loss, use_time_loss, use_mr_loss, use_mr_loss, use_mr_loss, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, g_mat, g_gan_t, d_real_t, d_fake_t, g_gan_mr, d_real_mr, d_fake_mr, d_real, d_fake):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,g_mat,g_gan_t,d_real_t,d_fake_t, g_gan_mr, d_real_mr, d_fake_mr, d_real, d_fake),flags) if f]
        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define mdct and imdct
        self.up_ratio = self.opt.hr_sampling_rate / self.opt.lr_sampling_rate
        self.window = kbdwin(self.opt.win_length).to(self.device)
        #self._dct = DCT_2N_native()
        self._mdct = MDCT4(n_fft=self.opt.n_fft, hop_length=self.opt.hop_length, win_length=self.opt.win_length, window=self.window, device=self.device)
        #self._idct = IDCT_2N_native()
        self._imdct = IMDCT4(n_fft=self.opt.n_fft, hop_length=self.opt.hop_length, win_length=self.opt.win_length, window=self.window, device=self.device)

        ##### define networks
        # Generator network
        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            if opt.use_hifigan_D:
                from .ParallelWaveGAN.parallel_wavegan.models.hifigan import HiFiGANMultiScaleMultiPeriodDiscriminator
                self.hifigan_D = HiFiGANMultiScaleMultiPeriodDiscriminator().to(self.device)
            if opt.use_time_D:
                self.time_D = networks.define_D(2, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, False, gpu_ids=self.gpu_ids)
            if opt.use_multires_D:
                self.multires_D = networks.define_MR_D(opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D,  gpu_ids=self.gpu_ids, base_nfft=2*opt.n_fft, window=kbdwin, min_value=opt.min_value, mdct_type='4')

        ### Encoder network
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder',
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)
            if opt.use_hifigan_D:
                self.load_network(self.hifigan_D, 'hifigan_D', opt.which_epoch, pretrained_path)
            if opt.use_time_D:
                self.load_network(self.time_D, 'time_D', opt.which_epoch, pretrained_path)
            if opt.use_multires_D:
                self.load_network(self.multires_D, 'multires_D', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            # pools that store previously generated samples
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, opt.use_match_loss, opt.use_hifigan_D or opt.use_time_D, opt.use_multires_D)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if opt.use_match_loss:
                self.criterionMatch = torch.nn.MSELoss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','G_mat','G_GAN_t','D_real_t','D_fake_t','G_GAN_mr','D_real_mr','D_fake_mr','D_real','D_fake')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))
            else:
                params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            print('Total number of parameters of G: %d' % (sum([param.numel() for param in params])))
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            if self.opt.use_hifigan_D:
                params += list(self.hifigan_D.parameters())
            if self.opt.use_time_D:
                params += list(self.time_D.parameters())
            print('Total number of parameters of D: %d' % (sum([param.numel() for param in params])))
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def to_spectro(self, audio, mask=False, mask_size=None):
        # Forward Transformation (MDCT)
        if self.opt.use_time_D:
            spectro, frames = self._mdct(audio.to(self.device),True)
            spectro = spectro.unsqueeze(1).permute(0,1,3,2)
        else:
            spectro = self._mdct(audio.to(self.device)).unsqueeze(1).permute(0,1,3,2)
            frames = None

        if self.opt.explicit_encoding:
            neg = 0.5*(torch.abs(spectro)-spectro)
            pos = spectro+neg
            log_spectro = torch.cat(
                (
                    aF.amplitude_to_DB(self.opt.alpha*pos+(1-self.opt.alpha)*neg, 20, self.opt.min_value, 1),
                    aF.amplitude_to_DB((1-self.opt.alpha)*pos+self.opt.alpha*neg, 20, self.opt.min_value, 1)
                ),
                dim=1,
            )
        elif self.opt.arcsinh_transform:
            Gain = self.opt.arcsinh_gain
            spectro = Gain*spectro
            log_spectro = torch.arcsinh(spectro)/torch.log(torch.Tensor([10])).to(self.device)
            #log_spectro = torch.cat((log_spectro, log_spectro.abs()), dim=1)
        else:
            log_spectro = aF.amplitude_to_DB(
            (torch.abs(spectro)+ self.opt.min_value),20,self.opt.min_value,1
            ).to(self.device)
        pha = torch.sign(spectro)

        mean = log_spectro.mean().float()
        std  = log_spectro.var().sqrt().float()
        audio_max = log_spectro.max(-1,True).values.max(-2,True).values.float()
        audio_min = log_spectro.min(-1,True).values.min(-2,True).values.float()

        #log_audio = (log_audio-mean)/std
        # Deprecated, for there already has been Instance Norm.

        # if explicit_encoding:
        #     # multiply phase with log magnitude
        #     log_audio = (log_audio-audio_min)/(audio_max-audio_min)
        #     # log_audio @ [0,1]
        #     log_audio = log_audio*pha
        #     # log_audio @ [-1,1], double peak
        if not self.opt.explicit_encoding:   #TODO
            if   self.opt.phase_encoding_mode == 'uni_dist':
                pha = pha*torch.rand(pha.size(), device=self.device)
            elif self.opt.phase_encoding_mode == 'norm_dist':
                _noise = torch.randn(pha.size(), device=self.device)
                _noise_min = _noise.min()
                _noise_max = _noise.max()
                _noise = (_noise - _noise_min)/(_noise_max - _noise_min)
                pha = pha*_noise
            elif self.opt.phase_encoding_mode == 'norm_dist2':
                _noise = torch.randn(pha.size(), device=self.device).abs()
                pha = pha*_noise
            elif self.opt.phase_encoding_mode == 'scale':
                pha = pha*0.5
        log_spectro = (log_spectro-audio_min)/(audio_max-audio_min)
        log_spectro = log_spectro*(self.opt.norm_range[1]-self.opt.norm_range[0])+self.opt.norm_range[0]
            # log_audio @ [-1,1], singal peak

        if mask:
            # mask the lr spectro so that it does not learn from garbage infomation
            size = log_spectro.size()
            if mask_size is None:
                mask_size = int(size[2]*(1-1/self.up_ratio))

            # fill the blank mask with noise
            _noise = torch.randn(size[0], size[1], mask_size, size[3], device=self.device)
            _noise_min = _noise.min()
            _noise_max = _noise.max()

            if self.opt.mask_mode == None or self.opt.fit_residual:
                _noise = torch.zeros(size[0], size[1], mask_size, size[3], device=self.device)
            elif self.opt.mask_mode == 'mode0':
                #fill empty with randn noise, single peak, centered at 0
                _noise = _noise/(_noise_max - _noise_min)
                #_noise @ [-1,1]
            elif self.opt.mask_mode == 'mode1':
                #fill empty with randn noise, double peak, mimic the real distribution
                _noise = (_noise - _noise_min)/(_noise_max - _noise_min)
                #_noise @ [0,1]
                psudo_pha = 2*torch.randint(low=0,high=2,size=_noise.size(), device=self.device)-1
                _noise = _noise * psudo_pha
                #_noise @ [-1,1]
            elif self.opt.mask_mode == 'mode2':
                #fill empty with randn noise, single peak, centered at 0.5
                _noise = (_noise - _noise_min)/(_noise_max - _noise_min)

            log_spectro = torch.cat(
                    (
                        log_spectro[:,:,:-mask_size,:],
                        _noise
                    ),dim=2)
        return log_spectro.float(), pha, {'max':audio_max, 'min':audio_min, 'mean':mean, 'std':std, 'frames':frames}

    def denormalize(self, log_spectro, norm_param):
        spectro = (log_spectro-self.opt.norm_range[0])/(self.opt.norm_range[1]-self.opt.norm_range[0])
        spectro = log_spectro*(norm_param['max'].to(self.device)-norm_param['min'].to(self.device))+norm_param['min'].to(self.device)
        #log_mag = log_mag*norm_param['std']+norm_param['mean']
        if self.opt.arcsinh_transform:
            return torch.sinh(spectro*torch.log(torch.Tensor([10])).to(self.device))/self.opt.arcsinh_gain
        else:
            return aF.DB_to_amplitude(spectro.to(self.device),10,0.5)-self.opt.min_value

    def to_audio(self, log_spectro, norm_param, pha=None):
        spectro = self.denormalize(log_spectro, norm_param)
        if self.opt.explicit_encoding:
            spectro = (spectro[...,0,:,:]-spectro[...,1,:,:])/(2*self.opt.alpha-1)
        elif self.opt.arcsinh_transform:
            pass
        else:
            if self.up_ratio > 1:
                size = pha.size(-2)
                psudo_pha = 2*torch.randint(low=0,high=2,size=pha.size(),device=self.device)-1
                pha = torch.cat((pha[...,:int(size*(1/self.up_ratio)),:],psudo_pha[...,int(size*(1/self.up_ratio)):,:]),dim=-2)
                spectro = spectro*pha

        if self.opt.explicit_encoding:
            audio = self._imdct(spectro.permute(0,2,1).contiguous())
        else:
            audio = self._imdct(spectro.squeeze(1).permute(0,2,1).contiguous())
        return audio

    def to_frames(self, log_spectro, norm_param):
        spectro = self.denormalize(log_spectro, norm_param)
        if self.opt.explicit_encoding:
            spectro = (spectro[...,0,:,:]-spectro[...,1,:,:])/(2*self.opt.alpha-1)
        _, frames = self._imdct(spectro.squeeze().permute(0,2,1).contiguous(), True)
        return frames

    def encode_input(self, lr_audio, inst_map=None, hr_audio=None, feat_map=None):
        # hires audio for training
        if hr_audio is not None:
            with torch.no_grad():
                hr_spectro, hr_pha, hr_norm_param = self.to_spectro(hr_audio, mask = self.opt.mask_hr, mask_size=int(self.opt.n_fft*(1-self.opt.sr_sampling_rate/self.opt.hr_sampling_rate)//2))
            #hr_spectro = Variable(hr_spectro.data.cuda())
        else:
            hr_spectro = None
            hr_pha = None
            hr_norm_param = None

        with torch.no_grad():
            lr_spectro, lr_pha, lr_norm_param = self.to_spectro(lr_audio, mask = True)
        #lr_spectro = lr_spectro.data.cuda()

        """ if self.opt.label_nc == 0:
            lr_spectro = lr_spectro.data.cuda()
        else:
            # create one-hot vector for label map
            size = lr_spectro.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, lr_spectro.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half() """
        # get edges from instance map (deprecated)

        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            #edge_map = self.get_edges(inst_map)
            lr_spectro = torch.cat((lr_spectro, inst_map), dim=1)
        #lr_spectro = Variable(lr_spectro, volatile=infer)

        # instance map for feature encoding (deprecated)
        """ if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())
             if self.opt.label_feat:
                #inst_map = label_map.cuda()
                inst_map = lr_pha.cuda() """

        return lr_spectro, lr_pha, hr_spectro, hr_pha, feat_map, inst_map, hr_norm_param, lr_norm_param

    def discriminate_F(self, input_label, test_image, use_pool=False):
        '''Frequency domain discriminator'''
        # notice the test_image is detached, hence it wont backward to G networks.
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def discriminate_time_D(self, label_spectro, test_spectro):
        '''Time domain discriminator'''
        # notice the test_image is detached, hence it wont backward to G networks.
        label_spectro = aF.amplitude_to_DB(torch.abs(label_spectro), 20, self.opt.min_value, 1)
        test_spectro = aF.amplitude_to_DB(torch.abs(test_spectro.detach()), 20, self.opt.min_value, 1)
        input_concat = torch.cat((label_spectro, test_spectro), dim=1)
        return self.time_D.forward(input_concat)

    def discriminate_hifi(self, input, norm_param=None, pha=None, is_spectro=True):
        '''Time domain discriminator using hifi_gan_D'''
        # input shape [B 1 T]
        if is_spectro:
            waveform = self.imdct(input, norm_param=norm_param, pha=pha).squeeze().unsqueeze(1)
        else:
            waveform = input.to(self.device).unsqueeze(1)
        return self.hifigan_D.forward(waveform)

    def forward(self, lr_audio, inst, hr_audio, feat, infer=False):
        # Encode Inputs
        lr_spectro, lr_pha, hr_spectro, hr_pha, feat_map, inst_map, hr_norm_param, lr_norm_param = self.encode_input(lr_audio, inst, hr_audio, feat)
        # if not self.opt.explicit_encoding and self.opt.input_nc>=2:
        #     lr_spectro = torch.cat((lr_spectro, lr_pha), dim=1)
        #     hr_spectro = torch.cat((hr_spectro, hr_pha), dim=1)
        # Fake Generation
        if self.use_features:
            if not self.opt.load_features:
                # for training
                # todo: implement multiple encoding method
                # use lr_pha temporaily. It will be replaced by inst_map for general propose
                feat_map = self.netE.forward(hr_spectro, lr_pha)
            # when inferrencing, it will select one from kmeans
            input_concat = torch.cat((lr_spectro, feat_map), dim=1)
        else:
            input_concat = lr_spectro
        sr_result = self.netG.forward(input_concat)
        if self.opt.fit_residual:
            sr_result = sr_result+lr_spectro

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate_F(lr_spectro, sr_result, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate_F(lr_spectro, hr_spectro)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        # make a new input pair without detaching, the loss will hence backward to G
        pred_fake = self.netD.forward(torch.cat((lr_spectro, sr_result), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Multi_Res Discriminator loss
        loss_G_GAN_mr = 0
        loss_D_real_mr = 0
        loss_D_fake_mr = 0
        if self.opt.use_multires_D:
            sr_audio = self.to_audio(sr_result, norm_param=lr_norm_param).squeeze(1)
            lr_audio = lr_audio.to(self.device).unsqueeze(1)
            hr_audio = hr_audio.to(self.device).unsqueeze(1)

            # Fake Detection and Loss
            pred_fake_pool_mr = self.multires_D.forward(torch.cat((lr_audio, sr_audio.detach()), dim=1))
            loss_D_fake_mr = self.criterionGAN(pred_fake_pool_mr, False)*self.opt.lambda_mr

            # Real Detection and Loss
            pred_real_mr = self.multires_D.forward(torch.cat((lr_audio, hr_audio), dim=1))
            loss_D_real_mr = self.criterionGAN(pred_real_mr, True)*self.opt.lambda_mr

            # GAN loss (Fake Passability Loss)
            # make a new input pair without detaching, the loss will hence backward to G
            pred_fake_mr = self.multires_D.forward(torch.cat((lr_audio, sr_audio), dim=1))
            loss_G_GAN_mr = self.criterionGAN(pred_fake_mr, True)*self.opt.lambda_mr

        # Time domain GAN, including hifi_gan_D and D on frames
        loss_G_GAN_time = 0
        loss_D_real_time = 0
        loss_D_fake_time = 0

        if self.opt.use_hifigan_D:
            pred_fake_time = self.discriminate_hifi(sr_result, norm_param=lr_norm_param, is_spectro=True)
            loss_G_GAN_time += self.criterionGAN(pred_fake_time, True)*self.opt.lambda_time
            pred_real_time = self.discriminate_hifi(hr_audio.detach(), is_spectro=False)
            loss_D_real_time += self.criterionGAN(pred_real_time, True)*self.opt.lambda_time
            _pred_fake_time = self.discriminate_hifi(sr_result.detach(), norm_param=lr_norm_param, is_spectro=True)
            loss_D_fake_time += self.criterionGAN(_pred_fake_time, False)*self.opt.lambda_time
        if self.opt.use_time_D:
            sr_frames = self.to_frames(sr_result,lr_norm_param).unsqueeze(1).to(torch.half if self.opt.fp16 else torch.float)
            lr_frames = lr_norm_param['frames'].unsqueeze(1)
            hr_frames = hr_norm_param['frames'].unsqueeze(1)

            pred_fake_frames = self.discriminate_time_D(lr_frames, sr_frames)
            loss_D_fake_time += self.criterionGAN(pred_fake_frames, False)*self.opt.lambda_time

            pred_real_frames = self.discriminate_time_D(lr_frames, hr_frames)
            loss_D_real_time += self.criterionGAN(pred_real_frames, True)*self.opt.lambda_time

            _pred_fake_frames = self.time_D.forward(torch.cat((lr_frames, sr_frames), dim=1))
            loss_G_GAN_time += self.criterionGAN(_pred_fake_frames, True)*self.opt.lambda_time


        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        # (deprecated) VGG feature matching loss
        loss_G_VGG = 0
        """ if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(sr_result, hr_spectro) * self.opt.lambda_feat """

        # Phase matching loss
        # If all phase are generated correctly, loss_G_pha will be 0
        loss_G_match = 0
        if self.opt.explicit_encoding:
            sr_pha = torch.sign(sr_result[:,0,:,:]-sr_result[:,1,:,:]).unsqueeze(1)
            if self.opt.use_match_loss:
                half = self.opt.win_length//2
                sr_frames = self.to_frames(sr_result,lr_norm_param)
                a = sr_frames[...,:-1,half:]*self.window[:half]
                b = sr_frames[...,1:,:half]*self.window[half:]
                loss_G_match = self.criterionMatch(a,b) * self.opt.lambda_mat

        # Register current samples
        if self.opt.arcsinh_transform:
            self.current_lable = (lr_spectro.detach().cpu()-self.opt.norm_range[0])/(self.opt.norm_range[1]-self.opt.norm_range[0])
            self.current_lable = (self.current_lable*(lr_norm_param['max'].cpu()-lr_norm_param['min'].cpu())+lr_norm_param['min'].cpu()).numpy()[0,0,:,:]

            min_val = hr_norm_param['min'].cpu().numpy()[0,0,:,:]
            max_val = hr_norm_param['max'].cpu().numpy()[0,0,:,:]
            self.current_generated = (sr_result.detach().cpu()-self.opt.norm_range[0])/(self.opt.norm_range[1]-self.opt.norm_range[0])
            self.current_generated = np.clip((self.current_generated*(lr_norm_param['max'].cpu()-lr_norm_param['min'].cpu())+lr_norm_param['min'].cpu()).numpy()[0,0,:,:], min_val, max_val)

            self.current_real = (hr_spectro.detach().cpu()-self.opt.norm_range[0])/(self.opt.norm_range[1]-self.opt.norm_range[0])
            self.current_real = (self.current_real*(hr_norm_param['max'].cpu()-hr_norm_param['min'].cpu())+hr_norm_param['min'].cpu()).numpy()[0,0,:,:]
        else:
            self.current_lable     = lr_spectro.detach().cpu().numpy()[0,0,:,:]
            self.current_generated = sr_result.detach().cpu().numpy()[0,0,:,:]
            self.current_real      = hr_spectro.detach().cpu().numpy()[0,0,:,:]

        # Additional visuals
        if self.opt.explicit_encoding:
            self.current_lable     = 0.5*(lr_spectro[0,0,:,:]+lr_spectro[0,1,:,:]).detach().cpu().numpy()
            self.current_generated = 0.5*(sr_result[0,0,:,:]+sr_result[0,1,:,:]).detach().cpu().numpy()
            self.current_real      = 0.5*(hr_spectro[0,0,:,:]+hr_spectro[0,1,:,:]).detach().cpu().numpy()
            if self.opt.input_nc>=2:
                self.current_lable_pha     = (hr_pha-sr_pha).detach().cpu().numpy()[0,0,:,:]
                self.current_generated_pha = sr_pha.detach().cpu().numpy()[0,0,:,:]
                self.current_real_pha      = hr_pha.detach().cpu().numpy()[0,0,:,:]
        else:
            self.current_lable_pha = None
            self.current_generated_pha = None
            self.current_real_pha = None

        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_match, loss_G_GAN_time, loss_D_real_time, loss_D_fake_time, loss_G_GAN_mr, loss_D_real_mr, loss_D_fake_mr, loss_D_real, loss_D_fake ), None if not infer else sr_result ]

    def inference(self, lr_audio, inst):
        # Encode Inputs
        lr_spectro, lr_pha, hr_spectro, hr_pha, feat_map, inst_map,hr_norm_param, lr_norm_param = self.encode_input(lr_audio, inst, None)

        with torch.no_grad():
            # Fake Generation
            if self.use_features:
                if self.opt.use_encoded_image:
                    # encode the real image to get feature map
                    feat_map = self.netE.forward(sr_spectro, inst_map)
                else:
                    # sample clusters from precomputed features
                    feat_map = self.sample_features(inst_map)
                input_concat = torch.cat((lr_spectro, feat_map), dim=1)
            else:
                input_concat = lr_spectro
            sr_spectro = self.netG.forward(input_concat)

        return sr_spectro, lr_pha, lr_norm_param, lr_spectro

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)
        if self.opt.use_hifigan_D:
            self.save_network(self.hifigan_D, 'hifigan_D', which_epoch, self.gpu_ids)
        if self.opt.use_time_D:
            self.save_network(self.time_D, 'time_D', which_epoch, self.gpu_ids)
        if self.opt.use_multires_D:
            self.save_network(self.multires_D, 'multires_D', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def get_current_visuals(self):
        lable_sp, lable_hist, _ = compute_visuals(sp=self.current_lable, abs=self.opt.abs_spectro)
        _, _, lable_pha = compute_visuals(pha=self.current_lable_pha)
        generated_sp, generated_hist, _ = compute_visuals(sp=self.current_generated, abs=self.opt.abs_spectro)
        _, _, generated_pha = compute_visuals(pha=self.current_generated_pha)
        real_sp, real_hist, _ = compute_visuals(sp=self.current_real, abs=self.opt.abs_spectro)
        _, _, real_pha = compute_visuals(pha=self.current_real_pha)
        if self.current_lable_pha is not None:
            return {'lable_spectro':        lable_sp,
                    'generated_spectro':    generated_sp,
                    'real_spectro':         real_sp,
                    'lable_hist':           lable_hist,
                    'generated_hist':       generated_hist,
                    'real_hist':            real_hist,
                    'lable_pha':            lable_pha,
                    'generated_pha':        generated_pha,
                    'real_pha':             real_pha}
        else:
            return {'lable_spectro':        lable_sp,
                    'generated_spectro':    generated_sp,
                    'real_spectro':         real_sp,
                    'lable_hist':           lable_hist,
                    'generated_hist':       generated_hist,
                    'real_hist':            real_hist,}

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)