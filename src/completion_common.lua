local common 
if paths.filep('common.lua') then 
  common = dofile('common.lua')
elseif paths.filep('../common.lua') then
  common = dofile('../common.lua')
elseif paths.filep('../../common.lua') then
  common = dofile('../../common.lua')
else
  error('could not find common.lua')
end

function common.step_to_vx_res(step) 
  if     step == 1 then return 64
  elseif step == 2 then return 128
  elseif step == 3 then return 256
  elseif step == 4 then return 512
  else   print(step); error('invalid step ') end
end


function common.eval_metrics(output, target, mask)
  local output = output:float()
  if oc.isOctree(output) then output = output:to_cdhw() end
  local target = target:float()
  if oc.isOctree(target) then target = target:to_cdhw() end
  if mask then mask = mask:float() end
  if mask and oc.isOctree(mask) then mask = mask:to_cdhw() end

  local sads = {}
  local rmses = {}
  local ious = {}
  local tps = torch.cmul(torch.le(output, 0), torch.le(target, 0)):float()
  local fps = torch.cmul(torch.le(output, 0), torch.gt(target, 0)):float()
  local fns = torch.cmul(torch.gt(output, 0), torch.le(target, 0)):float()
  local tns = torch.cmul(torch.gt(output, 0), torch.gt(target, 0)):float()
  if mask then
    tps:cmul(mask)
    fps:cmul(mask)
    fns:cmul(mask)
    tns:cmul(mask)
  end
  for bidx = 1, output:size(1) do
    local diff = (output[bidx] - target[bidx]):abs()
    local sad = diff:sum()
    local rmse = diff:pow(2):sum()
    if mask then 
      sad = sad / mask:sum()
      rmse = torch.sqrt( rmse / mask:sum() )
    else
      sad = sad / output:nElement()
      rmse = torch.sqrt( rmse / output:nElement() )
    end
    table.insert(sads, sad)
    table.insert(rmses, rmse)

    local tp = tps[bidx]:sum()
    local fp = fps[bidx]:sum()
    local fn = fns[bidx]:sum()
    local tn = tns[bidx]:sum()
    local denom = tp + fp + fn
    if denom > 0 then
      table.insert(ious, tp / (denom))
    else
      print(string.format('[WARNING] IoU denom = 0: tp=%f, fp=%f, fn=%f, tn=%f', tp, fp, fn, tn))
      table.insert(ious, 1)
    end
  end

  local mu = 0
  for _, sad in ipairs(sads) do mu = mu + sad end
  print(string.format('[INFO] mean SAD=%f', mu / #sads))
  local mu = 0
  for _, rmse in ipairs(rmses) do mu = mu + rmse end
  print(string.format('[INFO] mean RMSE=%f', mu / #rmses))
  local mu = 0
  for _, iou in ipairs(ious) do mu = mu + iou end
  print(string.format('[INFO] mean IoU=%f', mu / #ious))

  return sads, rmses, ious
end


function common.write_metrics_json(out_path, sads, rmses, ious, ta_paths, timings)
  local f = io.open(out_path, 'w')
  f:write('{\n')

  f:write('  "sads": [\n')
  for idx = 1, #sads do
    f:write(string.format('    %f', sads[idx]))
    if idx < #sads then 
      f:write(',\n')
    else
      f:write('\n  ],\n')
    end
  end

  f:write('  "rmses": [\n')
  for idx = 1, #rmses do
    f:write(string.format('    %f', rmses[idx]))
    if idx < #rmses then 
      f:write(',\n')
    else
      f:write('\n  ],\n')
    end
  end

  f:write('  "ious": [\n')
  for idx = 1, #ious do
    f:write(string.format('    %f', ious[idx]))
    if idx < #ious then 
      f:write(',\n')
    else
      f:write('\n  ],\n')
    end
  end

  f:write('  "timings": [\n')
  for idx = 1, #timings do
    f:write(string.format('    %f', timings[idx]))
    if idx < #timings then 
      f:write(',\n')
    else
      f:write('\n  ],\n')
    end
  end

  f:write('  "ta_paths": [\n')
  for idx = 1, #ta_paths do
    f:write(string.format('    "%s"', ta_paths[idx]))
    if idx < #ta_paths then 
      f:write(',\n')
    else
      f:write('\n  ]\n')
    end
  end

  f:write('}\n')
end


function common.save_output(opt, set, batch_idx, input, target, output, mask)
  local out_data = {}
  if torch.type(input) ~= 'table' then out_data['input'] = {input} else out_data['input'] = input end 
  if torch.type(target) ~= 'table' then out_data['target'] = {target} else out_data['target'] = target end 
  if torch.type(output) ~= 'table' then out_data['output'] = {output} else out_data['output'] = output end 
  if mask then
    if torch.type(mask) ~= 'table' then out_data['mask'] = {mask} else out_data['mask'] = mask end 
  end

  for name, data in pairs(out_data) do
    for data_idx, entry in ipairs(data) do
      if torch.isTensor(entry) then
        local hdf5 = require('hdf5')
        local out_path = paths.concat(opt.out_root, string.format('%s_set%02d_%02d_e%03d_b%03d.h5', name, set, data_idx, opt.epoch, batch_idx))
        print('  h5 to '..out_path)
        local f = hdf5.open(out_path, 'w')
        local options = hdf5.DataSetOptions()
        local out_entry = entry:float()
        f:write('data', out_entry, options)
        f:close()
      else
        local out_path = paths.concat(opt.out_root, string.format('%s_set%02d_%02d_e%03d_b%03d.oc', name, set, data_idx, opt.epoch, batch_idx))
        print('  oc to '..out_path)
        local o2d = oc.OctreeToCDHW():float()
        local out_entry = entry:float()
        out_entry:write_to_bin(out_path)
      end
    end
  end
end


function common.test_epoch(opt, data_loader, set)
  local set = set or 99
  local net = opt.net or error('no net in test_epoch')
  local criterion = opt.criterion or error('no criterion in test_epoch')
  local n_batches = data_loader:n_batches()

  net:evaluate()

  local avg_fs = {0}
  local sads = {}
  local rmses = {}
  local ious = {}
  local ta_paths = {}
  local timings = {}
  for batch_idx = 1, n_batches do
    print(string.format('[INFO] test batch %d/%d', batch_idx, n_batches))

    local timer = torch.Timer()
    local input, target, bin_paths, bta_paths = data_loader:getBatch()
    print(string.format('[INFO] loading data took %f[s]', timer:time().real))

    local mask = nil
    if opt.target_mask then
      if torch.type(target) ~= 'table' then error('target has to be a table') end
      if #target > 2 then
        local new_target = {}
        for tidx = 1, #target - 1 do table.insert(new_target, target[tidx]) end
        mask = target[#target]
        target = new_target
      else
        mask = target[2]
        target = target[1]
      end
    end
      
    local timer = torch.Timer()
    local output = net:forward(input)
    table.insert(timings, timer:time().real)
    if mask then
      output:cmul(mask)
      target:cmul(mask)
    end
    -- print(torch.type(input))
    -- print(torch.type(output))
    -- print(torch.type(target))
    -- print(input:size())
    -- print(output:size())
    -- print(target:size())
    -- print(input:min(), input:max())
    -- print(output:min(), output:max())
    -- print(target:min(), target:max())
    -- print(input:min(), input:max(), output:min(), output:max(), target:min(), target:max())

    local f = criterion:forward(output, target)

    print(string.format('[INFO] net/crtrn fwd took %f[s]', timer:time().real))
    if torch.type(output) == 'table' and criterion.criterions then
      for out_idx = 1, #output do 
        if batch_idx == 1 then avg_fs[out_idx] = 0 end
        avg_fs[out_idx] = avg_fs[out_idx] + criterion.criterions[out_idx].output
        print(string.format('  avg_f batch_idx=%d, out_idx=%d ... %f', batch_idx, out_idx, criterion.criterions[out_idx].output))
      end
    else
      avg_fs[1] = avg_fs[1] + f
      print(string.format('  avg_f batch_idx=%d, out_idx=%d ... %f', batch_idx, 1, f))
    end

    if opt.convert_output_fcn then 
      print('[INFO] convert target/output')
      target = opt.convert_output_fcn(target)
      output = opt.convert_output_fcn(output)
    end

    if opt.save_metrics then
      local bsads, brmses, bious = common.eval_metrics(output, target, mask)
      for _, sad in ipairs(bsads) do table.insert(sads, sad) end
      for _, rmse in ipairs(brmses) do table.insert(rmses, rmse) end
      for _, iou in ipairs(bious) do table.insert(ious, iou) end
      for _, ta_path in ipairs(bta_paths) do table.insert(ta_paths, ta_path) end
    end

    if opt.save_output then
      print('[INFO] write output')
      common.save_output(opt, set, batch_idx, input, target, output, mask)
    end

  end 

  if opt.save_metrics then
    local out_path = paths.concat(opt.out_root, string.format('metrics_e%04d_s%02d.json', opt.epoch, set))
    common.write_metrics_json(out_path, sads, rmses, ious, ta_paths, timings)
  end

  local avg_str = ''
  for idx, avg_f in ipairs(avg_fs) do
    avg_f = avg_f / n_batches
    avg_str = string.format('%s, avg_f[%d]=%f', avg_str, idx, avg_f)
  end
  if opt.save_metrics then
    avg_str = avg_str..', mu(MAE)='..common.table_mean(sads) 
    avg_str = avg_str..', mu(RMSE)='..common.table_mean(rmses) 
    avg_str = avg_str..', mu(IoU)='..common.table_mean(ious) 
    avg_str = avg_str..', mu(timing)='..common.table_mean(timings) 
  end
  print(string.format('Set %d | test_epoch=%d%s', set, opt.epoch, avg_str))
end


function common.clear_opt(opt, learningRate) 
  opt.weightDecay = 0.0001
  opt.learningRate = learningRate or 1e-4
  opt.learningRate_steps = {}
  opt.state_save_interval = 1
  opt.test_interval = 1
  opt.optimizer = optim['adam']
  -- clear epoch counter
  opt.epoch = nil
  -- clear optim fields
  opt.t = nil
  opt.m = nil
  opt.v = nil
  opt.t = nil
  opt.denom = nil
end


function common.get_best_epoch(root, set)
  local best_sad = 1e9
  local best_epoch = -1
  local metric_paths = common.walk_paths(root, 'json')
  metric_paths = common.match_paths(metric_paths, string.format('metrics_e.*_s%02d.json', set))
  for _, metric_path in ipairs(metric_paths) do
    local epoch = tonumber( string.sub(metric_path, -13, -10) )
    print(metric_path)
    local f = io.open(metric_path, 'r')
    local content = f:read('*a')
    f:close()
    local data = common.json:decode(content)
    local sads = data['sads']
    local sad = 0
    for _, val in ipairs(sads) do sad = sad + val end
    sad = sad / #sads
    if sad < best_sad then
      best_sad = sad
      best_epoch = epoch
    end
  end 
  return best_epoch, best_sad
end


function common.match_paths_(data_paths, prefix, encoding, setting, vx_res)
  local pattern = string.format('%s_.*_%s_%s_r%d.oc', prefix, encoding, setting, vx_res)
  local matched_paths = common.match_paths(data_paths, pattern) 
  return matched_paths
end 




function common.add_input_paths(opt, data_loader, data_paths, prefix, step, generated)
  local vx_res = common.step_to_vx_res(step)
  if generated then
    local input_paths = common.match_paths_(data_paths, string.format('gen%s', prefix), 'feat', opt.setting, vx_res)
    data_loader:addInputPaths(input_paths, opt.channels_in, vx_res, 'oc')
    print('[INFO] add '..#input_paths..' input paths - generated')
  else
    for _, encoding in ipairs(opt.encoding) do
      local mul 
      if encoding == 'tsdf' then
        mul = function(x) return x:mul(1.0 / common.step_to_truncation(step)) end
      end
      local load_type = 'oc'
      if step == 1 then
        load_type = 'oc2cdhw_cpu'
      end
      -- print(data_paths)
      local input_paths = common.match_paths_(data_paths, prefix, encoding, opt.setting, vx_res)
      print('[INFO] add '..#input_paths..' input paths of prefix "'..prefix..'", encoding "'..encoding..'" and setting "'..opt.setting..'" and vx_res "'..vx_res..'"')
      data_loader:addInputPaths(input_paths, opt.channels_in, vx_res, load_type, mul, nil)
    end
  end

  return data_loader
end

function common.get_train_data_loader(opt, data_paths, step, batch_size)
  local train_data_loader = common.DataLoader(batch_size, true) 
  common.add_input_paths(opt, train_data_loader, data_paths, 'train', step, step > 1)
  common.add_target_paths(opt, train_data_loader, data_paths, 'train', step)
  return train_data_loader
end

function common.get_test_data_loader(opt, data_paths, step, batch_size)
  local test_data_loader = common.DataLoader(batch_size, false) 
  common.add_input_paths(opt, test_data_loader, data_paths, 'test', step, step > 1)
  common.add_target_paths(opt, test_data_loader, data_paths, 'test', step)

  local val_data_loader = common.DataLoader(batch_size, false) 
  common.add_input_paths(opt, val_data_loader, data_paths, 'val', step, step > 1)
  common.add_target_paths(opt, val_data_loader, data_paths, 'val', step)

  test_data_loaders = {test_data_loader, val_data_loader}
  return test_data_loaders
end

function common.get_data_loader(opt, data_paths, step, batch_size)
  local train_data_loader = common.get_train_data_loader(opt, data_paths, step, batch_size)
  local test_data_loaders = common.get_test_data_loader(opt, data_paths, step, batch_size)
  return train_data_loader, test_data_loaders
end

function common.get_data_loader_for_create(opt, data_paths, step, prefix)
  local data_loader = common.DataLoader(1, false) 
  common.add_input_paths(opt, data_loader, data_paths, prefix, step, step > 1)
  common.add_input_paths(opt, data_loader, data_paths, prefix, step+1, false)
  common.add_target_paths(opt, data_loader, data_paths, prefix, step)

  return data_loader
end



function common.create_level_data(opt, step, data_paths, prefix)
  print('[INFO] save_level_data')
  paths.mkdir(opt.out_root)

  local vx_res = common.step_to_vx_res(step+1)

  local rec_mod = nn.Identity()
  local guide_mod = nn.Identity()
  if opt.no_guide_mod == true and step >= 2 then
    print('[INFO] using no guide mod!')
    guide_mod = nil
  end
  local split_mod = oc.OctreeSplitTsdf(rec_mod, guide_mod, 1):cuda()
  local cat_mod = oc.OctreeConcat(false, false, true):cuda()

  local data_loader = common.get_data_loader_for_create(opt, data_paths, step, prefix)
  data_loader.verbose = true
    
  local n_inputs = #opt.encoding
  if opt.mask_input then 
    n_inputs = n_inputs + 1
  end

  local mean_ratio = 0
  local n_batches = data_loader:n_batches()
  for batch_idx = 1, n_batches do
    local out_path = paths.concat(opt.out_root, string.format('gen%s_%08d_%s_%s_r%d.oc', prefix, batch_idx, 'feat', opt.setting, vx_res))
    table.insert(data_paths, out_path)
      
    if opt.net then
      print('  save batch to '..out_path)
      local input, target = data_loader:getBatch()
      if torch.type(input) ~= 'table' then input = {input} end
      local net_input = {}
      local cat_input = {}
      for input_idx = 1, #input do
        if input_idx <= (#input - n_inputs) then
          table.insert(net_input, input[input_idx])
        else
          table.insert(cat_input, input[input_idx])
        end
      end
      if #net_input == 1 then net_input = net_input[1] end
      
      local output = opt.net:forward(net_input)
      local rec = output

      -- split this level output and upsample
      local rec = rec_mod:forward(rec)
      if guide_mod then
        guide_mod:forward(cat_input[1])
      end
      local out = split_mod:forward(rec)

      -- combine next level input with this level input
      for _, cat_in in ipairs(cat_input) do
        out = cat_mod:forward({out, cat_in:to_cdhw()})
      end

      print('n_leafs     : '..out.grid.n_leafs)
      print('dense voxels: '..(vx_res*vx_res*vx_res))
      mean_ratio = mean_ratio + (out.grid.n_leafs)/(vx_res*vx_res*vx_res)

      out:write_to_bin(out_path)
    end
  end
  print('mean_ratio='..mean_ratio/n_batches)
  
  return data_paths
end

function common.test_biggest_sample(opt, data_loader, entry_idx)
  print('[INFO] test_biggest_sample')
  local max_path = ''
  local max_leafs = 0
  local max_path_idx = -1
  for path_idx, path in ipairs(data_loader.entries[entry_idx].paths) do
    print('  load header of '..path)
    local grid = oc.FloatOctree()
    grid:read_header_from_bin(path)
    if max_leafs < grid:n_leafs() then
      max_leafs = grid:n_leafs()
      max_path = path
      max_path_idx = path_idx
    end
  end 
  
  print(string.format('max leafs (%d): %s', max_leafs, max_path))
  local input, target = data_loader:getData(max_path_idx, max_path_idx)
  print('net forward')
  local output = opt.net:forward(input)
  print('criterion forward')
  local f = opt.criterion:forward(output, target[1])
  print('criterion backward')
  local dfdx = opt.criterion:backward(output, target[1])
  print('net backward')
  dfdx:cmul(target[2])
  opt.net:backward(input, dfdx)
  print('done')
end


function common.get_net_step(opt, step)
  local old_net = opt.net
  opt.net = get_net(#opt.encoding, opt.channels_in, opt.channels_out, step)
  if step > 1 then
    common.copy_params(old_net:float(), opt.net:float())
  end
  opt.net = opt.net:cuda()
  return opt
end


function common.ctf_loop(opt)
  local out_root = opt.out_root
  opt.save_metrics = true
  
  local data_paths = common.walk_paths(opt.ex_data_root)
  for step = 1, 3 do
    opt.out_root = paths.concat(out_root, 'step'..step)
    common.clear_opt(opt)
    
    opt.n_epochs = common.n_epochs_p_step[step]

    if step > 1 then
      opt.batch_size = 1
      opt.grad_iters = 4
    end
    
    local net_path = paths.concat(opt.out_root, string.format('net_epoch%03d.t7', opt.n_epochs))
    if paths.filep(net_path) then
      print('[INFO] load net for step='..step..' - '..net_path)

      -- determine best epoch
      local best_epoch, best_sad = common.get_best_epoch(opt.out_root, 2)
      print(string.format('[INFO] best_epoch=%d with sad=%f', best_epoch, best_sad))
      local net_path = paths.concat(opt.out_root, string.format('net_epoch%03d.t7', best_epoch))
      
      opt.net = torch.load(net_path)

    else
      print('[INFO] train for step='..(step))

      opt = common.get_net_step(opt, step) 
      
      local train_data_loader, test_data_loaders = common.get_data_loader(opt, data_paths, step, opt.batch_size)
      train_data_loader.verbose = true
      for _, test_data_loader in ipairs(test_data_loaders) do
        test_data_loader.verbose = true
      end
      
      common.worker(opt, train_data_loader, test_data_loaders)
    end

    -- check if gen files for next step exist, otherwise create them 
    local add_data_paths = common.walk_paths(opt.out_root)
    add_train_paths = common.match_paths_(add_data_paths, 'gentrain', 'feat', opt.setting, common.step_to_vx_res(step+1))
    add_test_paths = common.match_paths_(add_data_paths, 'gentest', 'feat', opt.setting, common.step_to_vx_res(step+1))
    add_val_paths = common.match_paths_(add_data_paths, 'genval', 'feat', opt.setting, common.step_to_vx_res(step+1))
    if #add_train_paths ~= 0 and #add_test_paths ~= 0 and #add_val_paths then
      print('[INFO] load gen data paths for step='..step)
      table.sort(add_train_paths)
      data_paths = common.table_combine(data_paths, add_train_paths)
      table.sort(add_test_paths)
      data_paths = common.table_combine(data_paths, add_test_paths)
      table.sort(add_val_paths)
      data_paths = common.table_combine(data_paths, add_val_paths)
    else
      print('[INFO] gen data for step='..(step))
      local best_epoch, best_sad = common.get_best_epoch(opt.out_root, 2)
      print(string.format('[INFO] best_epoch=%d with sad=%f for gen data', best_epoch, best_sad))
      local net_path = paths.concat(opt.out_root, string.format('net_epoch%03d.t7', best_epoch))
      opt.net = torch.load(net_path)

      -- data_paths = common.create_level_data(opt, step, data_paths, 'train')
      data_paths = common.create_level_data(opt, step, data_paths, 'test')
      data_paths = common.create_level_data(opt, step, data_paths, 'val')
    end
  end 
end


function common.evaluate_step(opt, step, epoch)
  local data_paths = common.walk_paths(opt.ex_data_root)
  local out_root = opt.out_root

  if step > 1 then
    opt.batch_size = 1
    opt.grad_iters = 4
  end

  for step = 1,3 do
    opt.out_root = paths.concat(out_root, 'step'..step)
    local add_data_paths = common.walk_paths(opt.out_root)

    add_train_paths = common.match_paths_(add_data_paths, 'gentrain', 'feat', opt.setting, common.step_to_vx_res(step+1))
    table.sort(add_train_paths)
    data_paths = common.table_combine(data_paths, add_train_paths)
    
    local add_test_paths = common.match_paths_(add_data_paths, 'gentest', 'feat', opt.setting, common.step_to_vx_res(step+1))
    table.sort(add_test_paths)
    data_paths = common.table_combine(data_paths, add_test_paths)
    
    local add_val_paths = common.match_paths_(add_data_paths, 'genval', 'feat', opt.setting, common.step_to_vx_res(step+1))
    table.sort(add_val_paths)
    data_paths = common.table_combine(data_paths, add_val_paths)
  end

  local out_epoch
  if epoch then 
    opt.out_root = paths.concat(out_root, 'step'..step)
    local net_path = paths.concat(opt.out_root, string.format('net_epoch%03d.t7', epoch))
    print('[INFO] load network '..net_path)
    opt.net = torch.load(net_path)
    out_epoch = epoch
  else
    opt.out_root = paths.concat(out_root, 'step'..step)
    local best_epoch_test, best_sad_test = common.get_best_epoch(opt.out_root, 1)
    print(string.format('[INFO] best epoch for step %d according to test set = %d (SAD=%f)', step, best_epoch_test, best_sad_test))
    local best_epoch_val, best_sad_val = common.get_best_epoch(opt.out_root, 2)
    print(string.format('[INFO] best epoch for step %d according to validation set = %d (SAD=%f)', step, best_epoch_val, best_sad_val))
    out_epoch = best_epoch_val
    if best_epoch_val == 0 then 
      local tmp_out_root = paths.concat(out_root, 'step'..(step-1))
      local best_epoch_test, best_sad_test = common.get_best_epoch(tmp_out_root, 1)
      print(string.format('[INFO] best epoch for step %d according to test set = %d (SAD=%f)', step-1, best_epoch_test, best_sad_test))
      local best_epoch_val2, best_sad_val = common.get_best_epoch(tmp_out_root, 2)
      print(string.format('[INFO] best epoch for step %d according to validation set = %d (SAD=%f)', step-1, best_epoch_val2, best_sad_val))
      if best_epoch_val2 == 0 then error('again epoch 0 is the best, something went wrong during training') end
      local net_path = paths.concat(tmp_out_root, string.format('net_epoch%03d.t7', best_epoch_val2))
      print('[INFO] load network '..net_path)
      opt.net = torch.load(net_path)
    else
      local net_path = paths.concat(opt.out_root, string.format('net_epoch%03d.t7', best_epoch_val))
      print('[INFO] load network '..net_path)
      opt.net = torch.load(net_path)
    end
  end

  local tmp_save_metrics = opt.save_metrics 
  local tmp_save_output = opt.save_output 
  local tmp_epoch = opt.epoch
  opt.save_metrics = false
  opt.save_output = true
  opt.epoch = out_epoch

  local train_data_loader, test_data_loaders = common.get_data_loader(opt, data_paths, step, opt.batch_size)
  for set, test_data_loader in ipairs(test_data_loaders) do
    common.test_epoch(opt, test_data_loader, set)
  end

  opt.save_metrics = tmp_save_metrics
  opt.save_output = tmp_save_output
  opt.epoch = tmp_epoch
end

return common
