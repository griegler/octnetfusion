#!/usr/bin/env th

local paths = require('paths')
local lfs = require('lfs')

local host = io.popen('uname -snr'):read('*line')
local user = io.popen('id -u -n'):read('*line')
local script_dir = paths.dirname(debug.getinfo(1, 'S').source:sub(2))
print(string.format('[LOAD CONFIG] hostname: "%s"', host))
print(string.format('[LOAD CONFIG] user: "%s"', user))
print(string.format('[LOAD CONFIG] script_dir: "%s"', script_dir))

common = {}


--------------------------------------------------------------------------------
-- helper functions
--------------------------------------------------------------------------------
function common.get_backend()
  local backend 
  if cudnn then 
    print('[INFO] USING CUDNN')
    backend = cudnn 
  else 
    print('[INFO] USING NN')
    backend = nn 
  end
  return backend
end

function common.walk_paths(root, ext)
  local function walk_paths_(d, ps) 
    for path in paths.iterdirs(d) do
      local path = paths.concat(d, path)
      walk_paths_(path, ps)
    end
    for path in paths.iterfiles(d) do
      local path = paths.concat(d, path)

      if ext then
        if ext == paths.extname(path) then
          table.insert(ps, path)
        end
      else 
        table.insert(ps, path)
      end
    end
  end 

  local ps = {}
  walk_paths_(root, ps)
  table.sort(ps)
  return ps
end

function common.paths_from_file(path)
  local ps = {}
  for p in io.lines(path) do
    table.insert(ps, p)
  end
  return ps
end 

function common.walk_paths_cached(root, ext)
  local cache_path 
  if ext then 
    cache_path = paths.concat(root, string.format('cache_paths_%s.txt', ext))
  else
    cache_path = paths.concat(root, string.format('cache_paths.txt'))
  end

  if paths.filep(cache_path) then
    print('[INFO] load paths from cache '..cache_path)
    return common.paths_from_file(cache_path)
  else
    print('[INFO] write paths to cache '..cache_path)
    local ps = common.walk_paths(root, ext)
    local f = io.open(cache_path, 'w')
    for _, p in ipairs(ps) do
      f:write(string.format('%s\n', p))
    end
    io.close(f)
    return ps
  end
end

function common.match_paths(ps, pattern)
  local rps = {}
  for _, p in ipairs(ps) do
    if p:match(pattern) then
      table.insert(rps, p)
    end
  end 
  table.sort(rps)
  return rps
end 

function common.table_shuffle(tab)
  local cnt = #tab
  while cnt > 1 do 
    local idx = math.random(cnt)
    tab[idx], tab[cnt] = tab[cnt], tab[idx]
    cnt = cnt - 1
  end
end 

function common.table_length(tab)
  local cnt = 0
  for _ in pairs(tab) do cnt = cnt + 1 end
  return cnt
end 

function common.table_clear(tab)
  for k in pairs(tab) do tab[k] = nil end
end

function common.table_combine(tab1, tab2)
  local tab_c = {}
  for idx = 1, #tab1 do
    table.insert(tab_c, tab1[idx])
  end
  for idx = 1, #tab2 do
    table.insert(tab_c, tab2[idx])
  end
  return tab_c
end 

function common.table_mean(tab)
  local mu = 0.0
  for idx, val in ipairs(tab) do
    mu = mu + val 
  end
  return mu / #tab
end

function common.string_split(str, sSeparator, nMax, bRegexp)
  assert(sSeparator ~= '')
  assert(nMax == nil or nMax >= 1)
  local aRecord = {}

  if str:len() > 0 then
    local bPlain = not bRegexp
    nMax = nMax or -1

    local nField, nStart = 1, 1
    local nFirst,nLast = str:find(sSeparator, nStart, bPlain)
    while nFirst and nMax ~= 0 do
      aRecord[nField] = str:sub(nStart, nFirst-1)
      nField = nField+1
      nStart = nLast+1
      nFirst,nLast = str:find(sSeparator, nStart, bPlain)
      nMax = nMax-1
    end
    aRecord[nField] = str:sub(nStart)
  end
  return aRecord
end

function common.octree_batch_to_cuda(ocf, occ)
  occ = ocf:cuda(occ)
end


function common.net_he_init(net)
  local function conv_init(model, name)
    for k,v in pairs(model:findModules(name)) do
      local n = v.kT * v.kW * v.kH * v.nOutputPlane
      v.weight:normal(0, math.sqrt(2/n))
      v.bias:zero()
    end 
  end 

  local function linear_init(model)
    for k, v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
    end
  end 

  conv_init(net, 'cudnn.VolumetricConvolution')
  conv_init(net, 'nn.VolumetricConvolution')
  conv_init(net, 'oc.OctreeConvolutionMM')
  linear_init(net)
end





local DataLoader = torch.class('common.DataLoader')

function DataLoader:__init(batch_size, train_mode, verbose)
  self.batch_size = batch_size or error('')
  self.train_mode = train_mode or false
  self.verbose = verbose or false
  self.data_idx = 0
  self.entries = {}
end

function DataLoader:addPaths(paths, channels, vx_size, ext, ptype, cpu_pp_fcn, gpu_pp_fcn)
  if ptype ~= 'input' and ptype ~= 'target' then
    print(ptype)
    error('unknown ptype')
  end
  if #self.entries > 0 and #self.entries[1].paths ~= #paths then
    print(#self.entries[1].paths)
    print(#paths)
    error('invalid number of paths')
  end
  self.n_samples = #paths

  local entry = {}
  entry.paths = paths or error('invalid paths')
  entry.channels = channels or error('invalid channels')
  entry.vx_size = vx_size or error('invalid vx_size')
  entry.ext = ext or error('invalid ext')
  entry.type = ptype
  entry.cpu_pp_fcn = cpu_pp_fcn or function(x) return x end
  entry.gpu_pp_fcn = gpu_pp_fcn or function(x) return x end
  table.insert(self.entries, entry)

end

function DataLoader:addInputPaths(paths, channels, vx_size, ext, cpu_pp_fcn, gpu_pp_fcn)
  self:addPaths(paths, channels, vx_size, ext, 'input', cpu_pp_fcn, gpu_pp_fcn)
end

function DataLoader:addTargetPaths(paths, channels, vx_size, ext, cpu_pp_fcn, gpu_pp_fcn)
  self:addPaths(paths, channels, vx_size, ext, 'target', cpu_pp_fcn, gpu_pp_fcn)
end

function DataLoader:shufflePaths()
  if self.verbose then
    print('[INFO] DataLoader: SHUFFLE SHUFFLE SHUFFLE')
  end
  local idx1 = #self.entries[1].paths
  while idx1 > 1 do 
    local idx2 = math.random(idx1)

    for entry_idx = 1, #self.entries do
      local tmp = self.entries[entry_idx].paths[idx2]
      self.entries[entry_idx].paths[idx2] = self.entries[entry_idx].paths[idx1]
      self.entries[entry_idx].paths[idx1] = tmp
    end 

    idx1 = idx1 - 1
  end
end

function DataLoader:clearState()
  for eidx = 1, #self.entries do
    self.entries[eidx].cpu_data = nil
    self.entries[eidx].gpu_data = nil
    self.entries[eidx].tmp_data = nil
    self.entries[eidx].oc2cdhw = nil
  end 
end 

function DataLoader:getData(from_idx, to_idx) 
  local in_data = {}
  local ta_data = {}
  local in_paths = {}
  local ta_paths = {}
  for eidx = 1, #self.entries do

    local batch_paths = {}
    for dix = from_idx, to_idx do
      local path = self.entries[eidx].paths[dix]
      if self.verbose then
        print(string.format('[INFO] DataLoader: load path %s', path))
      end
      if self.entries[eidx].ext == 'labels' or self.entries[eidx].ext == 'values' or paths.filep(path) then
        table.insert(batch_paths, path)
      else
        error('invalid path: '..path)
      end
    end
    
    if self.entries[eidx].ext == 'cdhw' then
      local bs = to_idx - from_idx + 1
      self.entries[eidx].cpu_data = torch.FloatTensor(bs, self.entries[eidx].channels, self.entries[eidx].vx_size, self.entries[eidx].vx_size, self.entries[eidx].vx_size)
      oc.read_dense_from_bin_batch(batch_paths, self.entries[eidx].cpu_data)
      self.entries[eidx].cpu_data = self.entries[eidx].cpu_pp_fcn(self.entries[eidx].cpu_data)
      self.entries[eidx].gpu_data = self.entries[eidx].gpu_data or torch.CudaTensor()
      self.entries[eidx].gpu_data:resize(self.entries[eidx].cpu_data:size())
      self.entries[eidx].gpu_data:copy(self.entries[eidx].cpu_data)
      self.entries[eidx].gpu_data = self.entries[eidx].gpu_pp_fcn(self.entries[eidx].gpu_data)

    elseif self.entries[eidx].ext == 'oc' then 
      self.entries[eidx].cpu_data = self.entries[eidx].cpu_data or oc.FloatOctree()
      self.entries[eidx].cpu_data:read_from_bin_batch(batch_paths)
      self.entries[eidx].cpu_data = self.entries[eidx].cpu_pp_fcn(self.entries[eidx].cpu_data)
      self.entries[eidx].gpu_data = self.entries[eidx].cpu_data:cuda(self.entries[eidx].gpu_data)
      self.entries[eidx].gpu_data = self.entries[eidx].gpu_pp_fcn(self.entries[eidx].gpu_data)

    elseif self.entries[eidx].ext == 'oc2cdhw_gpu' then
      self.entries[eidx].cpu_data = self.entries[eidx].cpu_data or oc.FloatOctree()
      self.entries[eidx].cpu_data:read_from_bin_batch(batch_paths)
      self.entries[eidx].cpu_data = self.entries[eidx].cpu_pp_fcn(self.entries[eidx].cpu_data)
      self.entries[eidx].tmp_data = self.entries[eidx].cpu_data:cuda(self.entries[eidx].tmp_data)
      self.entries[eidx].oc2cdhw = self.entries[eidx].oc2cdhw or oc.OctreeToCDHW():cuda()
      self.entries[eidx].gpu_data = self.entries[eidx].oc2cdhw:forward(self.entries[eidx].tmp_data)
      self.entries[eidx].gpu_data = self.entries[eidx].gpu_pp_fcn(self.entries[eidx].gpu_data)

    elseif self.entries[eidx].ext == 'oc2cdhw_cpu' then
      self.entries[eidx].cpu_data = self.entries[eidx].cpu_data or oc.FloatOctree()
      self.entries[eidx].cpu_data:read_from_bin_batch(batch_paths)
      self.entries[eidx].cpu_data = self.entries[eidx].cpu_pp_fcn(self.entries[eidx].cpu_data)
      self.entries[eidx].oc2cdhw = self.entries[eidx].oc2cdhw or oc.OctreeToCDHW()
      self.entries[eidx].tmp_data = self.entries[eidx].oc2cdhw:forward(self.entries[eidx].cpu_data)
      self.entries[eidx].gpu_data = self.entries[eidx].gpu_data or torch.CudaTensor()
      self.entries[eidx].gpu_data:resize(self.entries[eidx].tmp_data:size())
      self.entries[eidx].gpu_data:copy(self.entries[eidx].tmp_data)
      self.entries[eidx].gpu_data = self.entries[eidx].gpu_pp_fcn(self.entries[eidx].gpu_data)

    elseif self.entries[eidx].ext == 'labels' then
      self.entries[eidx].cpu_data = self.entries[eidx].cpu_data or torch.FloatTensor()
      self.entries[eidx].cpu_data:resize(#batch_paths)
      for bidx, batch_path in ipairs(batch_paths) do
        self.entries[eidx].cpu_data[bidx] = batch_paths[bidx]
      end
      self.entries[eidx].cpu_data = self.entries[eidx].cpu_pp_fcn(self.entries[eidx].cpu_data)
      self.entries[eidx].gpu_data = self.entries[eidx].gpu_data or torch.CudaTensor()
      self.entries[eidx].gpu_data:resize(self.entries[eidx].cpu_data:size())
      self.entries[eidx].gpu_data:copy(self.entries[eidx].cpu_data)
      self.entries[eidx].gpu_data = self.entries[eidx].gpu_pp_fcn(self.entries[eidx].gpu_data)
      -- print(self.entries[eidx].cpu_data)
    
    elseif self.entries[eidx].ext == 'values' then
      self.entries[eidx].cpu_data = self.entries[eidx].cpu_data or torch.FloatTensor()
      self.entries[eidx].cpu_data:resize(#batch_paths, self.entries[eidx].channels)
      for bidx, batch_path in ipairs(batch_paths) do
        for cidx = 1, self.entries[eidx].channels do
          self.entries[eidx].cpu_data[bidx][cidx] = batch_paths[bidx][cidx]
        end
      end
      self.entries[eidx].cpu_data = self.entries[eidx].cpu_pp_fcn(self.entries[eidx].cpu_data)
      self.entries[eidx].gpu_data = self.entries[eidx].gpu_data or torch.CudaTensor()
      self.entries[eidx].gpu_data:resize(self.entries[eidx].cpu_data:size())
      self.entries[eidx].gpu_data:copy(self.entries[eidx].cpu_data)
      self.entries[eidx].gpu_data = self.entries[eidx].gpu_pp_fcn(self.entries[eidx].gpu_data)
      -- print(self.entries[eidx].cpu_data)
      
    else
      error('unknown ext: '..self.entries[eidx].ext)
    end 


    if self.entries[eidx].type == 'input' then
      table.insert(in_data, self.entries[eidx].gpu_data)
      for bpidx = 1, #batch_paths do table.insert(in_paths, batch_paths[bpidx]) end
    elseif self.entries[eidx].type == 'target' then
      table.insert(ta_data, self.entries[eidx].gpu_data)
      for bpidx = 1, #batch_paths do table.insert(ta_paths, batch_paths[bpidx]) end
    else
      error('unknown type: '..self.entries[eidx].type)
    end
  end 
  if #in_data == 1 then in_data = in_data[1] end
  if #ta_data == 1 then ta_data = ta_data[1] end

  return in_data, ta_data, in_paths, ta_paths
end

function DataLoader:getBatch()
  local bs = math.min(self.batch_size, self.n_samples - self.data_idx)

  if self.data_idx == 0 and self.train_mode then
    self:shufflePaths()
  end
  
  local in_data, ta_data, in_paths, ta_paths = self:getData(self.data_idx + 1, self.data_idx + bs)

  self.data_idx = self.data_idx + bs
  if (self.train_mode and (self.n_samples - self.data_idx) < self.batch_size) or 
     (not self.train_mode and self.data_idx >= self.n_samples) then 
    self.data_idx = 0 
  end

  collectgarbage(); collectgarbage()
  return in_data, ta_data, in_paths, ta_paths
end

function DataLoader:size()
  return self.n_samples
end

function DataLoader:n_batches()
  if self.train_mode then
    return math.floor(self.n_samples / self.batch_size)
  else
    return math.ceil(self.n_samples / self.batch_size)
  end
end






function common.train_epoch(opt, data_loader)
  local net = opt.net or error('no net in train_epoch')
  local criterion = opt.criterion or error('no criterion in train_epoch')
  local optimizer = opt.optimizer or error('no optimizer in train_epoch')
  local n_batches = data_loader:n_batches()
  local grad_iters = opt.grad_iters or 1

  net:training()

  local parameters, grad_parameters = net:getParameters()
  local batch_idx = 1
  while batch_idx <= n_batches do
    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      grad_parameters:zero()

      local f = 0
      local grad_iter = 1
      while grad_iter <= grad_iters and batch_idx <= n_batches do
        local input, target = data_loader:getBatch()

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

        local output = net:forward(input)

        if mask then 
          output:cmul(mask)
          target:cmul(mask)
        end
        f = f + criterion:forward(output, target)
        local dfdx = criterion:backward(output, target)

        if mask then 
          dfdx:cmul(mask)
        end
        net:backward(input, dfdx)

        batch_idx = batch_idx + 1
        grad_iter = grad_iter + 1
      end

      if grad_iters > 1 then
        -- print('GRAD ITER ADJUST')
        f = f / (grad_iter - 1.0)
        grad_parameters:mul(1.0 / (grad_iter - 1))
      end
      
      if batch_idx < 129 or batch_idx % math.floor((n_batches / 200)) == 0 then 
        print(string.format('epoch=%2d | iter=%4d/%d | loss=%9.6f ', opt.epoch, batch_idx, n_batches, f))
      end
      
      return f, grad_parameters
    end 
    optimizer(feval, parameters, opt)
    xlua.progress(batch_idx, n_batches)
  end 
end

function common.test_epoch(opt, data_loader, set)
  local set = set or 99
  local net = opt.net or error('no net in test_epoch')
  local criterion = opt.criterion or error('no criterion in test_epoch')
  local n_batches = data_loader:n_batches()

  net:evaluate()

  local avg_f = 0
  for batch_idx = 1, n_batches do
    print(string.format('[INFO] test batch %d/%d', batch_idx, n_batches))

    local timer = torch.Timer()
    local input, target = data_loader:getBatch()
    print(string.format('[INFO] loading data took %f[s] - n_batches %d', timer:time().real, target:size(1)))

    local timer = torch.Timer()
    local output = net:forward(input)
    output = output[{{1,target:size(1)}, {}}]
    local f = criterion:forward(output, target)
    print(string.format('[INFO] net/crtrn fwd took %f[s]', timer:time().real))
    avg_f = avg_f + f
  end 
  avg_f = avg_f / n_batches

  print(string.format('Set %d | test_epoch=%d, avg_f=%f', set, opt.epoch, avg_f))
end

function common.copy_params(src, dst)
  local src_weights = src:parameters()
  local dst_weights = dst:parameters()
  for widx, src_weight in ipairs(src_weights) do
    local dst_weight = dst_weights[widx]
    local src_sz = src_weight:size()
    local dst_sz = dst_weight:size()
    if #src_sz == #dst_sz then
      local dim_eq = true
      for didx = 1, #src_sz do dim_eq = dim_eq and src_sz[didx] == dst_sz[didx] end
      if dim_eq then
        print(string.format('[INFO] copy params from %d/%d to %d/%d', widx,#src_weights, widx,#dst_weights))
        dst_weight:copy(src_weight)
      end
    end
  end
end

function common.network_memory_consumption(net)
  local function mem_dense(str)
    local mem = 0
    for _, mod in ipairs(net:findModules(str)) do
      if mod.output then mem = mem + 4 * mod.output:nElement() end
      if mod.weight then mem = mem + 4 * mod.weight:nElement() end
      if mod.bias then mem = mem + 4 * mod.bias:nElement() end
    end
    return mem 
  end 
  local function mem_oc(str)
    local mem = 0
    for _, mod in ipairs(net:findModules(str)) do
      mem = mem + mod.output:mem_using()
      if mod.weight then mem = mem + 4 * mod.weight:nElement() end
      if mod.bias then mem = mem + 4 * mod.bias:nElement() end
    end
    return mem
  end
  local mem = 0
  mem = mem + mem_dense('cudnn.VolumetricConvolution')
  mem = mem + mem_dense('cudnn.VolumetricAveragePooling')
  mem = mem + mem_dense('cudnn.VolumetricMaxPooling')
  mem = mem + mem_dense('nn.VolumetricConvolution')
  mem = mem + mem_dense('nn.VolumetricAveragePooling')
  mem = mem + mem_dense('nn.VolumetricMaxPooling')
  mem = mem + mem_oc('oc.OctreeConvolutionMM')
  mem = mem + mem_oc('oc.OctreeGridPool2x2x2')
  mem = mem + mem_oc('oc.OctreeToCDHW()')
  -- mem = mem + mem_dense('nn.Linear')
  return mem
end


function common.worker(opt, train_data_loader, test_data_loaders)
  -- create out root dir
  print(string.format('out_root: %s', opt.out_root))
  paths.mkdir(opt.out_root)

  -- enable logging
  local cmd = torch.CmdLine()
  cmd:log(paths.concat(opt.out_root, string.format('train_%d.log', sys.clock())))

  -- load state if it exists
  local state_path = paths.concat(opt.out_root, 'state.t7')
  if paths.filep(state_path) then 
    print('[INFO] LOADING PREVIOUS STATE')
    local opt_state = torch.load(state_path)
    opt_state.do_stats = opt.do_stats
    opt_state.save_output = opt.save_output
    for k, v in pairs(opt_state) do 
      if k ~= 'criterion' then
        opt[k] = v
      end
    end
  end

  local start_epoch = 1
  if opt.epoch then
    start_epoch = opt.epoch + 1
  end
    
  -- -- test network
  opt.epoch = start_epoch - 1
  for set, test_data_loader in ipairs(test_data_loaders) do
    common.test_epoch(opt, test_data_loader, set)
  end

  print(string.format('[INFO] start_epoch=%d', start_epoch))
  for epoch = start_epoch, opt.n_epochs do
    opt.epoch = epoch
    
    -- clean up
    opt.net:clearState()
    collectgarbage('collect')
    collectgarbage('collect')

    -- train
    print('[INFO] train epoch '..epoch..', lr='..opt.learningRate)
    opt.data_fcn = opt.train_data_fcn
    local timer = torch.Timer()
    common.train_epoch(opt, train_data_loader)
    print(string.format('[INFO] train epoch took %f[s]', timer:time().real))
     
    -- save network
    print('[INFO] save net')
    local net_path = paths.concat(opt.out_root, string.format('net_epoch%03d.t7', opt.epoch))
    torch.save(net_path, opt.net:clearState())
    print('[INFO] saved net to: ' .. net_path)

    -- save state
    if not opt.state_save_interval or opt.epoch % opt.state_save_interval == 0 then
      print('[INFO] save state')
      opt.net = opt.net:clearState()
      torch.save(state_path, opt)
      print('[INFO] saved state to: ' .. state_path)
    end

    -- clean up 
    collectgarbage('collect')
    collectgarbage('collect')

    -- adjust learning rate
    if opt.learningRate_steps[epoch] ~= nil then
      opt.learningRate = opt.learningRate * opt.learningRate_steps[epoch]
    end

    -- test network
    for set, test_data_loader in ipairs(test_data_loaders) do
      common.test_epoch(opt, test_data_loader, set)
    end
  end
    
end

return common
