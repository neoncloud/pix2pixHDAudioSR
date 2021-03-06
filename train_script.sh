python train.py --name mdct_explicit_phase_coding_mode0 --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 10 --gpu_id 0 --nThreads 0 --explicit_encoding --mask --mask_mode mode0

python train.py --name mdct_explicit_phase_coding_mode1 --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 10 --gpu_id 1 --nThreads 0 --explicit_encoding --mask --mask_mode mode1

python train.py --name mdct_implicit_phase_coding --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 10 --gpu_id 2 --nThreads 0 --mask --instance_feat --feat_num 1

python train.py --name mdct_implicit_phase_coding_mask0 --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 10 --gpu_id 3 --nThreads 0 --mask --mask_mode mode0 --instance_feat --feat_num 1

python train.py --name mdct_pretrain --dataroot /home/neoncloud/openslr/LibriSpeech/files.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 8 --gpu_id 0 --nThreads 8 --mask --mask_mode mode2

python generate_audio.py --name mdct_nophase2 --phase test --dataroot /root/VCTK-Corpus/wav48/p225/p225_003.wav --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 2 --serial_batches --nThreads 0 --mask --mask_mode mode2 --load_pretrain ./checkpoints/mdct_nophase2

python train.py --name mdct_2048 --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 8 --gpu_id 0 --nThreads 8 --mask --mask_mode mode0 --n_fft 2048 --win_length 2048

python train.py --name mdct_hifitts_pretrain --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 30 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --segment_length 25500

python train.py --name mdct_VCTK_with_pretrain --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 8 --gpu_id 2 --nThreads 16 --mask --mask_mode mode2 --segment_length 25500 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_local --continue_train

python train.py --name mdct_VCTK_with_pretrain_glob --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 30 --gpu_id 1 --nThreads 16 --mask --mask_mode mode2 --segment_length 25500 --load_pretrain ./checkpoints/mdct_hifitts_pretrain --niter 50 --niter_decay 50

python train.py --name mdct_hifitts_pretrain_amp --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --fp16 --validation_split 0.01 --abs_spectro --center

python generate_audio.py --name mdct_hifitts_pretrain_amp_gen --dataroot /root/VCTK-Corpus/wav48/p227/p227_004.wav --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 2 --serial_batches --nThreads 0 --mask --mask_mode mode2 --netG local --validation_split 0 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_amp2 --gpu_id 2 --center --phase test --serial_batches

python train.py --name mdct_hifitts_pretrain_test --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 32 --gpu_id 2 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --fp16 --validation_split 0.01 --abs_spectro --center --eval_freq 400 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_amp

python train.py --name mdct_VCTK_with_pretrain_amp --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_amp2 --continue_train --fp16 --validation_split 0.01 --abs_spectro --center

python train.py --name mdct_VCTK_with_pretrain_amp_2x --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 32 --gpu_id 1 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_amp2 --continue_train --fp16 --validation_split 0.01 --abs_spectro --center --lr_sampling_rate 24000

python train.py --name mdct_VCTK_with_pretrain_amp_3x --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_amp2 --continue_train --fp16 --validation_split 0.01 --abs_spectro --center --lr_sampling_rate 16000

python train.py --name mdct_VCTK_with_pretrain_amp_4x --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 32 --gpu_id 1 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_amp2 --continue_train --fp16 --validation_split 0.01 --abs_spectro --center --lr_sampling_rate 12000

tar -zcvf mdct_VCTK_with_pretrain_amp_6x.tgz /root/pix2pixHD/checkpoints/mdct_VCTK_with_pretrain_amp && tar -zcvf mdct_VCTK_with_pretrain_amp_2x.tgz /root/pix2pixHD/checkpoints/mdct_VCTK_with_pretrain_amp_2x && tar -zcvf mdct_VCTK_with_pretrain_amp_3x.tgz /root/pix2pixHD/checkpoints/mdct_VCTK_with_pretrain_amp_3x && tar -zcvf mdct_VCTK_with_pretrain_amp_4x.tgz /root/pix2pixHD/checkpoints/mdct_VCTK_with_pretrain_amp_4x && tar -zcvf mdct_hifitts_pretrain_amp2.tgz /root/pix2pixHD/checkpoints/mdct_hifitts_pretrain_amp2

python train.py --name mdct_hifitts_pretrain_explict_pha2 --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding
#G: 730,713,346 D: 5,531,522

python train.py --name mdct_VCTK_with_pretrain_explict_pha_6x --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 1 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_explict_pha2 --continue_train --fp16 --validation_split 0.01 --abs_spectro --center --lr_sampling_rate 8000 --explicit_encoding

python train.py --name mdct_VCTK_with_pretrain_explict_pha_4x --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_explict_pha2 --continue_train --fp16 --validation_split 0.01 --abs_spectro --center --lr_sampling_rate 12000 --explicit_encoding

python train.py --name mdct_VCTK_with_pretrain_explict_pha_3x --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_explict_pha2 --continue_train --fp16 --validation_split 0.01 --abs_spectro --center --lr_sampling_rate 16000 --explicit_encoding

python train.py --name mdct_VCTK_with_pretrain_explict_pha_2x --dataroot /root/VCTK-Corpus/wav48 --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 1 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --load_pretrain ./checkpoints/mdct_hifitts_pretrain_explict_pha2 --continue_train --fp16 --validation_split 0.01 --abs_spectro --center --lr_sampling_rate 24000 --explicit_encoding

## ablation study 
## 75,501,568 per n_blocks_global
python train.py --name mdct_hifitts_pha2_G7L3 --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 20 --niter_decay 10 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 7 --n_blocks_local 3
#G: 579710210  D: 5531522
python train.py --name mdct_hifitts_pha2_G5L3 --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 1 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 10 --niter_decay 0 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 5 --n_blocks_local 3
#G: 428707074 D: 5531522
python train.py --name mdct_hifitts_pha2_G3L2 --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 3 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 10 --niter_decay 0 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2
# G: 277408770 D: 5531522
# 295,168 per n_blocks_local

python train.py --name mdct_hifitts_pha2_G3L2_48ngf --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 3 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 10 --niter_decay 0 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 48
# G: 156050690 D: 5531522

python train.py --name mdct_hifitts_pha2_G3L2_32ngf --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 3 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 10 --niter_decay 0 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 32
# G: 69363202 D: 5531522

python train.py --name mdct_hifitts_pha2_G3L2_24ngf --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 2 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 10 --niter_decay 0 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 24
# G: 39020930 D: 5531522

python train.py --name mdct_hifitts_pha2_G3L2_16ngf --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 10 --niter_decay 0 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 16
# G: 17346306 D: 5531522

python train.py --name mdct_hifitts_pha2_G3L2_8ngf --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 2 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 10 --niter_decay 0 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 8
# G: 4339330 D: 5531522

python generate_audio.py --name pha2_G3L2_48_2x_gen --dataroot /root/VCTK-Corpus/wav48/p227/p227_004.wav --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 4 --serial_batches --nThreads 0 --mask --mask_mode mode2 --netG local --validation_split 0 --load_pretrain ./checkpoints/hifitts_vctk_pha2_G3L2_48ngf_2x --gpu_id 2 --center --phase test --serial_batches --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 48 --lr_sampling_rate 24000


python train.py --name mdct_hifitts_phaloss_G3L2_48ngf_6x --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 64 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 50 --niter_decay 50 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 48 --use_match_loss --save_epoch_freq 40 --save_latest_freq 2000 && python train.py --name hifitts_vctk_phaloss_G3L2_48ngf_6x --dataroot /root/VCTK-Corpus/train.csv --load_pretrain ./checkpoints/mdct_hifitts_phaloss_G3L2_48ngf_6x --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 64 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 100 --niter_decay 50 --fp16 --validation_split 0 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 48 --use_match_loss --save_epoch_freq 40 --save_latest_freq 2000

python eval_matric.py --name eval_hifitts_vctk_phaloss_G3L2_48ngf_6x --dataroot /root/VCTK-Corpus/test.csv --load_pretrain ./checkpoints/hifitts_vctk_phaloss_G3L2_48ngf_6x --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 64 --gpu_id 2 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 10 --validation_split 0 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 48

#wav48/p248/p248_011.wav
python generate_audio.py --name gen_hifitts_vctk_phaloss_G3L2_48ngf_6x --dataroot /root/VCTK-Corpus/wav48/p225/p225_002.wav --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 4 --serial_batches --nThreads 0 --mask --mask_mode mode2 --netG local --validation_split 0 --load_pretrain ./checkpoints/hifitts_vctk_phaloss_G3L2_48ngf_6x --gpu_id 2 --center --phase test --serial_batches --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 48

python train.py --name VCTK_G3L2_48ngf --dataroot /root/VCTK-Corpus/wav48 --load_pretrain ./checkpoints/hifitts_vctk_pha2_G3L2_48ngf_6x --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 64 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 50 --niter_decay 50 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 48 --eval_freq 5000 --save_latest_freq 2000 --save_epoch_freq 20 --use_match_loss

python generate_audio.py --name GEN_VCTK_G3L2_48ngf --dataroot /root/VCTK-Corpus/wav48/p225/p225_003.wav --load_pretrain ./checkpoints/VCTK_G3L2_48ngf --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 4 --gpu_id 1 --nThreads 0 --serial_batches --mask --mask_mode mode2 --netG local --validation_split 0 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 48 --phase test

python train.py --name hifitts_G3L2_48ngf_time_loss --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 3 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 100 --niter_decay 50 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 48 --eval_freq 5000 --save_latest_freq 2000 --save_epoch_freq 20 --use_hifigan_D

python train.py --name VCTK_G3L2_48ngf_match_loss_mse --dataroot /root/VCTK-Corpus/wav48 --load_pretrain ./checkpoints/hifitts_vctk_pha2_G3L2_48ngf_6x --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 50 --niter_decay 50 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 48 --eval_freq 5000 --save_latest_freq 2000 --save_epoch_freq 20 --use_match_loss

python generate_audio.py --name GEN_VCTK_G3L2_48ngf_match_loss_mse --dataroot ./test/test.wav --load_pretrain ./checkpoints/VCTK_G3L2_48ngf_match_loss_mse --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 4 --gpu_id 3 --nThreads 0 --serial_batches --mask --mask_mode mode2 --netG local --validation_split 0 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 48 --phase test

python train.py --name hifitts_G3L2_48ngf_time_D --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 32 --gpu_id 1 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 100 --niter_decay 50 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 48 --eval_freq 5000 --save_latest_freq 2000 --save_epoch_freq 20 --use_time_D --continue_train --lambda_time 10
#hifitts_G3L2_48ngf_time_D

python train.py --name VCTK_hifitts_G3L2_48ngf_time_D_match --dataroot /root/VCTK-Corpus/train.csv --load_pretrain ./checkpoints/VCTK_hifitts_G3L2_48ngf_time_D --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 64 --gpu_id 0 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 50 --niter_decay 0 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 3 --n_blocks_local 2 --ngf 48 --eval_freq 5000 --save_latest_freq 2000 --save_epoch_freq 20 --use_time_D --continue_train --lambda_time 10 --use_match_loss

python train.py --name VCTK_hifitts_G5L3_48ngf_log_time_D_4 --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 64 --gpu_id 3 --nThreads 16 --mask --mask_mode mode2 --netG local --niter 30 --niter_decay 20 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 5 --n_blocks_local 3 --ngf 48 --eval_freq 8000 --save_latest_freq 8000 --save_epoch_freq 20 --use_time_D --lambda_time 4

python train.py --name VCTK_hifitts_G5L3_48ngf_MR_D --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 72 --gpu_id 3 --nThreads 32 --mask --mask_mode mode2 --netG local --niter 30 --niter_decay 20 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 5 --n_blocks_local 3 --ngf 48 --eval_freq 8000 --save_latest_freq 8000 --save_epoch_freq 10 --lambda_feat 0.01

python train.py --name VCTK_hifitts_G5L3_48ngf_alpha1 --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 2 --input_nc 2 --batchSize 72 --gpu_id 2 --nThreads 32 --mask --netG local --niter 30 --niter_decay 20 --fp16 --validation_split 0.01 --abs_spectro --center --explicit_encoding --n_blocks_global 5 --n_blocks_local 3 --ngf 48 --eval_freq 8000 --save_latest_freq 8000 --save_epoch_freq 10 --alpha 0.9

python train.py --name VCTK_hifitts_G5L3_48ngf_arcsinh --dataroot /root/VCTK-Corpus/train.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 96 --gpu_id 2 --fp16 --nThreads 32 --mask --netG local --niter 30 --niter_decay 20 --validation_split 0.01 --center --arcsinh_transform --n_blocks_global 5 --n_blocks_local 3 --ngf 48 --eval_freq 8000 --save_latest_freq 8000 --save_epoch_freq 10 --abs_spectro --mask_mode mode2

python train.py --name VCTK_hifitts_G5L3_48ngf_arcsinh_timeD --dataroot /root/VCTK-Corpus/train.csv --load_pretrain ./checkpoints/VCTK_hifitts_G5L3_48ngf_arcsinh  --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 96 --gpu_id 2 --fp16 --nThreads 32 --mask --netG local --niter 100 --niter_decay 50 --validation_split 0.01 --center --arcsinh_transform --n_blocks_global 5 --n_blocks_local 3 --ngf 48 --eval_freq 8000 --save_latest_freq 8000 --save_epoch_freq 10 --abs_spectro --mask_mode mode2 --use_time_D

python train.py --name hifitts_G5L3_48ngf_arcsinh --dataroot /root/hi_fi_tts_v0/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 160 --gpu_id 2 --fp16 --nThreads 32 --mask --netG local --niter 30 --niter_decay 20 --validation_split 0 --center --arcsinh_transform --n_blocks_global 5 --n_blocks_local 3 --ngf 48 --eval_freq 8000 --save_latest_freq 8000 --save_epoch_freq 10 --abs_spectro --arcsinh_gain 200 --add_noise --snr 50

python train.py --name aishell_G5L3_48ngf_arcsinh_timeD --dataroot /root/aishell_3/train/audio.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 96 --gpu_id 2 --fp16 --nThreads 32 --mask --netG local --niter 30 --niter_decay 20 --validation_split 0 --center --arcsinh_transform --n_blocks_global 5 --n_blocks_local 3 --ngf 48 --eval_freq 8000 --save_latest_freq 8000 --save_epoch_freq 10 --abs_spectro --mask_mode mode2

python train.py --name hifitts_G5L3_48ngf_arcsinh_fitres2 --dataroot /root/hi_fi_tts_v0/clean.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 128 --gpu_id 2 --fp16 --nThreads 32 --mask --netG local --niter 70 --niter_decay 30 --validation_split 0.01 --center --arcsinh_transform --n_blocks_global 5 --n_blocks_local 3 --ngf 48 --eval_freq 8000 --save_latest_freq 8000 --save_epoch_freq 10 --abs_spectro --arcsinh_gain 500 --add_noise --snr 55 --norm_range -1 1 --lr 1.5e-4 --fit_residual

python train.py --name VCTK_hifitts_G5L3_48ngf_arcsinh_fitres --dataroot /root/VCTK-Corpus/train.csv --load_pretrain ./checkpoints/hifitts_G5L3_48ngf_arcsinh_fitres2 --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 128 --gpu_id 2 --fp16 --nThreads 16 --mask --netG local --niter 100 --niter_decay 100 --validation_split 0.01 --center --arcsinh_transform --n_blocks_global 5 --n_blocks_local 3 --ngf 48 --eval_freq 8000 --save_latest_freq 8000 --save_epoch_freq 10 --abs_spectro --arcsinh_gain 500 --norm_range -1 1 --lr 1.5e-4 --fit_residual

python train.py --name VCTK_hifitts_G5L3_48ngf_arcsinh_fitres_match --dataroot /root/VCTK-Corpus/train.csv --load_pretrain ./checkpoints/VCTK_hifitts_G5L3_48ngf_arcsinh_fitres --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 128 --gpu_id 2 --fp16 --nThreads 16 --mask --netG local --niter 100 --niter_decay 50 --validation_split 0.01 --center --arcsinh_transform --n_blocks_global 5 --n_blocks_local 3 --ngf 48 --eval_freq 8000 --save_latest_freq 8000 --save_epoch_freq 10 --abs_spectro --arcsinh_gain 500 --norm_range -1 1 --lr 1.5e-4 --fit_residual --use_match_loss

# working but too slow to train
python train.py --name VCTK_hifitts_G5L3_48ngf_arcsinh_fitres_hifigan --dataroot /root/VCTK-Corpus/train.csv --load_pretrain ./checkpoints/VCTK_hifitts_G5L3_48ngf_arcsinh_fitres --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 32 --gpu_id 1 --fp16 --nThreads 16 --mask --netG local --niter 100 --niter_decay 50 --validation_split 0.01 --center --arcsinh_transform --n_blocks_global 5 --n_blocks_local 3 --ngf 48 --eval_freq 8000 --save_latest_freq 8000 --save_epoch_freq 10 --abs_spectro --arcsinh_gain 500 --norm_range -1 1 --lr 1.5e-4 --fit_residual --use_hifigan_D

python train.py --name VCTK_hifitts_G5L3_48ngf_arcsinh_fitres_timeD2 --dataroot /root/VCTK-Corpus/train.csv --load_pretrain ./checkpoints/VCTK_hifitts_G5L3_48ngf_arcsinh_fitres --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 128 --gpu_id 2 --fp16 --nThreads 16 --mask --netG local --niter 100 --niter_decay 50 --validation_split 0.01 --center --arcsinh_transform --n_blocks_global 5 --n_blocks_local 3 --ngf 48 --eval_freq 8000 --save_latest_freq 8000 --save_epoch_freq 10 --abs_spectro --arcsinh_gain 500 --norm_range -1 1 --lr 1.5e-4 --fit_residual --use_time_D

python train.py --name hifitts_G5L3_48ngf_arcsinh_fitres2_interp --dataroot /root/hi_fi_tts_v0/clean.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 128 --gpu_id 2 --fp16 --nThreads 16 --mask --netG local --niter 70 --niter_decay 30 --validation_split 0.01 --center --arcsinh_transform --n_blocks_global 5 --n_blocks_local 3 --ngf 48 --eval_freq 16000 --save_latest_freq 16000 --save_epoch_freq 10 --abs_spectro --arcsinh_gain 500 --add_noise --snr 55 --norm_range -1 1 --lr 1.5e-4 --fit_residual --use_match_loss --upsample_type interpolate

python train.py --name VCTK_G4L3_48ngf_arcsinh_fitres_interp_attn --dataroot /mnt/e/VCTK-Corpus/train.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 8 --gpu_id 0 --fp16 --nThreads 16 --mask --netG local --niter 70 --niter_decay 30 --validation_split 0.01 --center --arcsinh_transform --n_blocks_global 4 --n_blocks_local 3 --ngf 48 --eval_freq 16000 --save_latest_freq 16000 --save_epoch_freq 10 --abs_spectro --arcsinh_gain 500 --norm_range -1 1 --lr 1.5e-4 --fit_residual --use_match_loss --upsample_type interpolate --n_blocks_attn 1

python train.py --name VCTK_G4L3_48ngf_arcsinh_fitres_interp_attn_multires --dataroot /mnt/e/VCTK-Corpus/train.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 8 --gpu_id 0 --fp16 --nThreads 16 --mask --netG local --niter 70 --niter_decay 30 --validation_split 0.01 --center --arcsinh_transform --n_blocks_global 4 --n_blocks_local 3 --ngf 48 --eval_freq 16000 --save_latest_freq 16000 --save_epoch_freq 10 --abs_spectro --arcsinh_gain 500 --norm_range -1 1 --lr 1.5e-4 --fit_residual --use_match_loss --upsample_type interpolate --n_blocks_attn 1 --use_multires_D --num_D 3

python train.py --name VCTK_G4A1L2A1_48ngf_arcsinh --dataroot /mnt/e/VCTK-Corpus/train.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 8 --gpu_id 0 --fp16 --nThreads 16 --mask --netG local --niter 70 --niter_decay 30 --validation_split 0.01 --center --arcsinh_transform --n_blocks_global 4 --n_blocks_local 3 --ngf 48 --eval_freq 16000 --save_latest_freq 16000 --save_epoch_freq 10 --abs_spectro --arcsinh_gain 500 --norm_range -1 1 --lr 1.5e-4 --fit_residual --use_match_loss --upsample_type interpolate --n_blocks_attn_l 1 --n_blocks_attn_g 1
python train.py --name VCTK_G4L3_48ngf_arcsinh_fitres_interp_attn --dataroot /root/VCTK-Corpus/train.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 128 --gpu_id 0 --fp16 --nThreads 16 --mask --netG local --niter 70 --niter_decay 30 --validation_split 0.01 --center --arcsinh_transform --n_blocks_global 4 --n_blocks_local 3 --ngf 48 --eval_freq 16000 --save_latest_freq 16000 --save_epoch_freq 10 --abs_spectro --arcsinh_gain 500 --norm_range -1 1 --lr 1.5e-4 --fit_residual --use_match_loss --upsample_type interpolate --n_blocks_attn 1

python train.py --name VCTK_G4L3A1_48ngf_arcsinh_fitres_interp_attn_multires3 --dataroot /root/VCTK-Corpus/train.csv --no_instance --no_vgg_loss --label_nc 0 --output_nc 1 --input_nc 1 --batchSize 128 --gpu_id 0 --fp16 --nThreads 4 --mask --netG local --niter 70 --niter_decay 30 --validation_split 0.01 --center --arcsinh_transform --n_blocks_global 4 --n_blocks_local 3 --ngf 48 --eval_freq 16000 --save_latest_freq 16000 --save_epoch_freq 5 --abs_spectro --arcsinh_gain 500 --norm_range -1 1 --fit_residual --upsample_type interpolate --n_blocks_attn 1 --use_multires_D --lambda_mr 0.4

python train.py \
--name VCTK_G4A3L3A0_48ngf_arcsinh_resconv_interp \
--dataroot /mnt/e/VCTK-Corpus/train.csv --batchSize 16 --validation_split 0.01 \
--label_nc 0 --output_nc 1 --input_nc 1 --gpu_id 1 --fp16 --nThreads 4 \
--mask --no_instance --no_vgg_loss --center --arcsinh_transform \
--abs_spectro --arcsinh_gain 500 --norm_range -1 1 \
--netG local --n_blocks_global 4 --n_blocks_local 3 --ngf 48 \
--n_blocks_attn_g 2 --n_blocks_attn_l 0 \
--fit_residual --upsample_type interpolate --downsample_type resconv \
--niter 150 --niter_decay 50 \
--eval_freq 16000 --save_latest_freq 16000 --save_epoch_freq 20

python train.py \
--name VCTK_G0A3L0A2_48ngf_arcsinh_resconv_interp \
--dataroot /home/neoncloud/VCTK-Corpus/train.csv --batchSize 16 --validation_split 0.01 \
--label_nc 0 --output_nc 1 --input_nc 1 --gpu_id 0 --fp16 --nThreads 4 \
--mask --no_instance --no_vgg_loss --center --arcsinh_transform \
--abs_spectro --arcsinh_gain 500 --norm_range -1 1 \
--netG local --n_blocks_global 0 --n_blocks_local 0 --ngf 48 \
--n_blocks_attn_g 3 --dim_head_g 128 --heads_g 4\
--n_blocks_attn_l 2 --dim_head_l 64 --heads_l 8\
--fit_residual --upsample_type interpolate --downsample_type conv \
--niter 70 --niter_decay 30 \
--eval_freq 16000 --save_latest_freq 16000 --save_epoch_freq 3 --print_freq 16

python generate_audio.py \
--phase test --load_pretrain ./checkpoints/VCTK_G4A3L3A0_48ngf_arcsinh_resconv_interp \
--name GEN_VCTK_G4A3L3A0_48ngf_arcsinh_resconv_interp \
--dataroot /home/neoncloud/VCTK-Corpus/wav48/p225/p225_002.wav --batchSize 2 --validation_split 0 \
--label_nc 0 --output_nc 1 --input_nc 1 --gpu_id 0 --fp16 --nThreads 1 \
--mask --no_instance --no_vgg_loss --center --arcsinh_transform \
--abs_spectro --arcsinh_gain 500 --norm_range -1 1 \
--netG local --n_blocks_global 4 --n_blocks_local 3 --ngf 48 \
--n_blocks_attn_g 1 --n_blocks_attn_l 0 \
--fit_residual --upsample_type interpolate --downsample_type resconv

python generate_audio.py \
--phase test --load_pretrain ./checkpoints/VCTK_G0A5_8h_48ngf_arcsinh_resconv_interp --name GEN_VCTK_G0A5_8h_48ngf_arcsinh_resconv_interp \
--dataroot /home/neoncloud/VCTK-Corpus/wav48/p225/p225_003.wav --batchSize 2 --validation_split 0.00 \
--label_nc 0 --output_nc 1 --input_nc 1 --gpu_id 0 --fp16 --nThreads 1 \
--mask --no_instance --no_vgg_loss --center --arcsinh_transform \
--abs_spectro --arcsinh_gain 500 --norm_range -1 1 \
--netG global --n_blocks_global 0 --n_blocks_local 0 --ngf 48 \
--n_blocks_attn_g 5 --dim_head_g 128 --heads_g 8 --proj_factor_g 4 \
--fit_residual --upsample_type interpolate --downsample_type resconv

#Total number of parameters of G: 13221569
python train.py \
--name VCTK_G0A4L2_8h_32ngf_mrD_nomask_conv_interp \
--dataroot /home/neoncloud/VCTK-Corpus/train.csv --batchSize 64 --validation_split 0.01 \
--label_nc 0 --output_nc 1 --input_nc 1 --gpu_id 0 --fp16 --nThreads 4 \
--no_instance --no_vgg_loss --center --arcsinh_transform \
--abs_spectro --arcsinh_gain 600 --norm_range -1 1 \
--netG local --n_blocks_global 0 --n_blocks_local 2 --ngf 32 \
--n_blocks_attn_g 4 --dim_head_g 32 --heads_g 8 --proj_factor_g 4 \
--fit_residual --upsample_type interpolate --downsample_type conv \
--use_multires_D --lambda_mr 0.3 --num_D 2 --num_mr_D 3 --ndf 48 \
--niter 70 --niter_decay 30 \
--eval_freq 16000 --save_latest_freq 16000 --save_epoch_freq 3 --print_freq 240

python generate_audio.py --name GEN_VCTK_G0A4L2_8h_32ngf_mrD_nomask_conv_interp \
--phase test --load_pretrain ./checkpoints/VCTK_G0A4L2_8h_32ngf_mrD_nomask_conv_interp \
--dataroot /home/neoncloud/VCTK-Corpus/wav48/p225/p225_003.wav --batchSize 2 --validation_split 0.00 \
--label_nc 0 --output_nc 1 --input_nc 1 --gpu_id 0 --fp16 --nThreads 2 \
--no_instance --no_vgg_loss --center --arcsinh_transform \
--abs_spectro --arcsinh_gain 600 --norm_range -1 1 \
--netG local --n_blocks_global 0 --n_blocks_local 2 --ngf 32 \
--n_blocks_attn_g 4 --dim_head_g 32 --heads_g 8 --proj_factor_g 4 \
--fit_residual --upsample_type interpolate --downsample_type conv

python train.py \
--name VCTK_G0A4L0A2_8h4h_16ngf_mrD_arcsinh_resconv_interp \
--dataroot /home/neoncloud/VCTK-Corpus/train.csv --batchSize 60 --validation_split 0.01 \
--label_nc 0 --output_nc 1 --input_nc 1 --gpu_id 0 --fp16 --nThreads 4 \
--no_instance --no_vgg_loss --center --arcsinh_transform \
--abs_spectro --arcsinh_gain 800 --norm_range -1 1 --smooth 0.5 \
--netG local --n_blocks_global 0 --n_blocks_local 0 --ngf 16 \
--n_blocks_attn_g 4 --dim_head_g 128 --heads_g 4 --proj_factor_g 4 \
--n_blocks_attn_l 4 --dim_head_l 64 --heads_l 8 --proj_factor_g 4 \
--fit_residual --upsample_type interpolate --downsample_type resconv \
--use_multires_D --lambda_mr 0.4 --num_D 3 --ndf 64 \
--niter 70 --niter_decay 30 \
--eval_freq 16000 --save_latest_freq 16000 --save_epoch_freq 3 --print_freq 240

python train.py \
--name VCTK_G0A4L2_8h_16ngf_mrD_arcsinh_resconv_interp \
--dataroot /home/neoncloud/audio-super-res/data/vctk/VCTK-Corpus/train.csv --batchSize 96 --validation_split 0.01 \
--label_nc 0 --output_nc 1 --input_nc 1 --gpu_id 0 --fp16 --nThreads 4 \
--no_instance --no_vgg_loss --center --arcsinh_transform \
--abs_spectro --arcsinh_gain 800 --norm_range -1 1 --smooth 0.5 \
--netG local --n_blocks_global 0 --n_blocks_local 2 --ngf 16 \
--n_blocks_attn_g 4 --dim_head_g 64 --heads_g 8 --proj_factor_g 4 \
--fit_residual --upsample_type interpolate --downsample_type resconv \
--use_multires_D --lambda_mr 0.4 --num_D 3 --ndf 64 \
--niter 70 --niter_decay 30 \
--eval_freq 16000 --save_latest_freq 16000 --save_epoch_freq 3 --print_freq 768

python generate_audio.py \
--name gen_hifitts_G4A3L3_6h_56ngf_arcsinh_resconv_interp \
--load_pretrain ./checkpoints/hifitts_G4A3L3_6h_56ngf_arcsinh_resconv_interp \
--dataroot /home/neoncloud/VCTK-Corpus/wav48/p225/p225_003.wav --batchSize 2 \
--validation_split 0.00 --label_nc 0 --output_nc 1 --input_nc 2 \
--gpu_id 0 --fp16 --nThreads 4 \
--no_instance --no_vgg_loss --center --arcsinh_transform --add_noise --snr 55 \
--abs_spectro --arcsinh_gain 500 --norm_range -1 1 --smooth 0.0 --netG local \
--n_downsample_global 3 --n_blocks_global 4 --n_blocks_local 3 --ngf 56 \
--n_blocks_attn_g 3 --dim_head_g 128 --heads_g 6 --proj_factor_g 4 \
--n_blocks_attn_l 0 --fit_residual --upsample_type interpolate --downsample_type resconv --phase test

#vctk_hifitts_G4A3L3_56ngf_6x
python generate_audio.py \
--name gen_vctk_hifitts_G4A3L3_56ngf_6x \
--load_pretrain ./checkpoints/vctk_hifitts_G4A3L3_56ngf_6x \
--dataroot /home/neoncloud/VCTK-Corpus/wav48/p225/p225_003.wav --batchSize 2 \
--validation_split 0.00 --label_nc 0 --output_nc 1 --input_nc 2 \
--gpu_id 0 --fp16 --nThreads 4 \
--no_instance --no_vgg_loss --center --arcsinh_transform --add_noise --snr 55 \
--abs_spectro --arcsinh_gain 500 --norm_range -1 1 --smooth 0.0 --netG local \
--n_downsample_global 3 --n_blocks_global 4 --n_blocks_local 3 --ngf 56 \
--n_blocks_attn_g 3 --dim_head_g 128 --heads_g 6 --proj_factor_g 4 \
--n_blocks_attn_l 0 --fit_residual --upsample_type interpolate --downsample_type resconv --phase test

#vctk_hifitts_G4A3L3_56ngf_6x
python generate_audio.py \
--name gen_vctk_hifitts_G4A3L3_56ngf_3x \
--load_pretrain ./checkpoints/vctk_hifitts_G4A3L3_56ngf_3x \
--dataroot /home/neoncloud/VCTK-Corpus/wav48/p225/p225_003.wav --batchSize 2 \
--validation_split 0.00 --label_nc 0 --output_nc 1 --input_nc 2 \
--gpu_id 0 --fp16 --nThreads 4 \
--no_instance --no_vgg_loss --center --arcsinh_transform --add_noise --snr 55 \
--abs_spectro --arcsinh_gain 500 --norm_range -1 1 --smooth 0.0 --netG local \
--n_downsample_global 3 --n_blocks_global 4 --n_blocks_local 3 --ngf 56 \
--n_blocks_attn_g 3 --dim_head_g 128 --heads_g 6 --proj_factor_g 4 \
--n_blocks_attn_l 0 --fit_residual --upsample_type interpolate --downsample_type resconv --phase test --lr_sampling_rate 16000

python train.py \
--name vctk_hifitts_G4A3L3_56ngf_6x_4 \
--load_pretrain ./checkpoints/vctk_hifitts_G4A3L3_56ngf_6x_3 \
--dataroot ~/VCTK-Corpus/train.csv --batchSize 32 \
--validation_split 0.01 --label_nc 0 --output_nc 1 --input_nc 2 \
--gpu_id 0 --fp16 --nThreads 4 \
--no_instance --no_vgg_loss --center \
--arcsinh_transform --add_noise --snr 60 \
--abs_spectro --arcsinh_gain 1000 \
--norm_range -1 1 --smooth 0.0 --abs_norm --src_range -5 5 \
--netG local --ngf 56 \
--n_downsample_global 3 --n_blocks_global 4 \
--n_blocks_attn_g 3 --dim_head_g 128 --heads_g 6 --proj_factor_g 4 \
--n_blocks_attn_l 0 --n_blocks_local 3 \
--fit_residual --upsample_type interpolate --downsample_type resconv \
--niter 50 --niter_decay 50 \
--use_multires_D --lambda_mr 0.6 \
--eval_freq 16000 --save_latest_freq 16000 --save_epoch_freq 4