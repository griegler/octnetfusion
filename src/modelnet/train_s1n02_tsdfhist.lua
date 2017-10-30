#!/usr/bin/env th

local common = dofile('../modelnet_fusion_common.lua')
require('nn')
require('cunn')
require('cudnn')
require('oc')
require('optim')
require('nngraph')

local opt = {}
opt.ex_data_root = paths.concat('preprocessed')
opt.encoding = {'tsdf_hist'}
opt.channels_in = 10
opt.channels_out = 1
opt.setting = 's1n02'
dofile('../unetadd.lua')
opt.out_root = paths.concat('results', 'encodings', 'tsdfhist')
opt.batch_size = 4
opt.cpu_ram_gb = 8
opt.gpu_ram_gb = 8

opt.criterion = nn.AbsCriterion()
opt.criterion = opt.criterion:cuda()

common.n_epochs_p_step = {50,25,25}
common.ctf_loop(opt)

common.evaluate_step(opt, 3)
