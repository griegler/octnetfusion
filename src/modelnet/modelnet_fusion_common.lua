local common 
if paths.filep('../completion_common.lua') then 
  common = dofile('../completion_common.lua')
elseif paths.filep('../../completion_common.lua') then 
  common = dofile('../../completion_common.lua')
elseif paths.filep('../../../completion_common.lua') then 
  common = dofile('../../../completion_common.lua')
else
  error('could not find completion_common.lua')
end

common.n_epochs_p_step = {20,20,20}

function common.step_to_truncation(step)
  local truncation_p_256 = 0.025
  if     step == 1 then return truncation_p_256 * 4
  elseif step == 2 then return truncation_p_256 * 2
  elseif step == 3 then return truncation_p_256 * 1
  elseif step == 4 then return truncation_p_256 * 0.5
  else   print(step); error('invalid step ') end
end


function common.add_target_paths(opt, data_loader, data_paths, prefix, step)
  local vx_res = common.step_to_vx_res(step)
  local mul = function(x) return x:mul(1.0 / common.step_to_truncation(step)) end
  local target_paths = common.match_paths_(data_paths, prefix, 'tsdf', 's0', vx_res)
  print('[INFO] add '..#target_paths..' target paths')
  data_loader:addTargetPaths(target_paths, opt.channels_out, vx_res, 'oc2cdhw_cpu', mul, nil)
end


return common
