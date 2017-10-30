function get_base_net(step)
  if step == 1 then
    local input = nn.Identity()()

    local enc1 = cudnn.VolumetricConvolution( 16,32, 3,3,3, 1,1,1, 1,1,1)(input)
    local enc1 = cudnn.ReLU(true)(enc1)

    local enc2 = cudnn.VolumetricMaxPooling(2,2,2)(enc1)
    local enc2 = cudnn.VolumetricConvolution( 32,32, 3,3,3, 1,1,1, 1,1,1)(enc2)
    local enc2 = cudnn.ReLU(true)(enc2)
    local enc2 = cudnn.VolumetricConvolution( 32,64, 3,3,3, 1,1,1, 1,1,1)(enc2)
    local enc2 = cudnn.ReLU(true)(enc2)
   
    local enc3 = cudnn.VolumetricMaxPooling(2,2,2)(enc2)
    local enc3 = cudnn.VolumetricConvolution( 64,64, 3,3,3, 1,1,1, 1,1,1)(enc3)
    local enc3 = cudnn.ReLU(true)(enc3)
    local enc3 = cudnn.VolumetricConvolution( 64,64, 3,3,3, 1,1,1, 1,1,1)(enc3)
    local enc3 = cudnn.ReLU(true)(enc3)
    local enc3 = cudnn.VolumetricConvolution( 64,64, 3,3,3, 1,1,1, 1,1,1)(enc3)
    local enc3 = cudnn.ReLU(true)(enc3)

    local dec2 = oc.VolumetricNNUpsampling(2,2,2)(enc3)
    local dec2 = nn.JoinTable(2)({dec2, enc2})
    local dec2 = cudnn.VolumetricConvolution(128,32, 3,3,3, 1,1,1, 1,1,1)(dec2)
    local dec2 = cudnn.ReLU(true)(dec2)
    local dec2 = cudnn.VolumetricConvolution( 32,32, 3,3,3, 1,1,1, 1,1,1)(dec2)
    local dec2 = cudnn.ReLU(true)(dec2)

    local dec1 = oc.VolumetricNNUpsampling(2,2,2)(dec2)
    local dec1 = nn.JoinTable(2)({dec1, enc1})
    local dec1 = cudnn.VolumetricConvolution(64,16, 3,3,3, 1,1,1, 1,1,1)(dec1)
    local dec1 = cudnn.ReLU(true)(dec1)
    local dec1 = cudnn.VolumetricConvolution( 16,16, 3,3,3, 1,1,1, 1,1,1)(dec1)
    local dec1 = cudnn.ReLU(true)(dec1)

    local net  = nn.gModule({input}, {dec1})
    return net
  else
    local n_grids = 0
    if step == 3 then
      n_grids = 64
      -- n_grids = 13824
    end

    local input = nn.Identity()()
    
    local mod_enc1 = oc.OctreeReLU(true)
    local mod_enc2 = oc.OctreeReLU(true)

    local enc1 = oc.OctreeConvolutionMM( 16,32, n_grids)(input)
    local enc1 = mod_enc1(enc1)

    local enc2 = oc.OctreeGridPool2x2x2('max')(enc1)
    local enc2 = oc.OctreeConvolutionMM( 32,32, n_grids)(enc2)
    local enc2 = oc.OctreeReLU(true)(enc2)
    local enc2 = oc.OctreeConvolutionMM( 32,64, n_grids)(enc2)
    local enc2 = mod_enc2(enc2)
   
    local enc3 = oc.OctreeGridPool2x2x2('max')(enc2)
    local enc3 = oc.OctreeConvolutionMM( 64,64, n_grids)(enc3)
    local enc3 = oc.OctreeReLU(true)(enc3)
    local enc3 = oc.OctreeConvolutionMM( 64,64, n_grids)(enc3)
    local enc3 = oc.OctreeReLU(true)(enc3)
    local enc3 = oc.OctreeConvolutionMM( 64,64, n_grids)(enc3)
    local enc3 = oc.OctreeReLU(true)(enc3)

    local dec2 = oc.OctreeGridUnpoolGuided2x2x2(mod_enc2)(enc3)
    local add_dec2 = oc.OctreeConcat()({dec2, enc2})
    local dec2 = add_dec2({dec2, enc2})
    local dec2 = oc.OctreeConvolutionMM(128,32, n_grids)(dec2)
    local dec2 = oc.OctreeReLU(true)(dec2)
    local dec2 = oc.OctreeConvolutionMM( 32,32, n_grids)(dec2)
    local dec2 = oc.OctreeReLU(true)(dec2)

    local dec1 = oc.OctreeGridUnpoolGuided2x2x2(mod_enc1)(dec2)
    local add_dec1 = oc.OctreeConcat()({dec1, enc1})
    add_dec1.output = oc.FloatOctree()
    local dec1 = add_dec1({dec1, enc1})
    local dec1 = oc.OctreeConvolutionMM(64,16, n_grids)(dec1)
    local dec1 = oc.OctreeReLU(true)(dec1)
    local dec1 = oc.OctreeConvolutionMM( 16,16, n_grids)(dec1)
    local dec1 = oc.OctreeReLU(true)(dec1)

    local net  = nn.gModule({input}, {dec1})
    return net
  end
end

function get_net(n_inputs, channels_in, channels_out, step)
  local base_net = get_base_net(step)
  if step == 1 then
    local input 
    if n_inputs == 1 then
      input = nn.Identity()()
    else
      input = nn.JoinTable(1,4)()
    end
    local enc1 = cudnn.VolumetricConvolution( channels_in,16, 3,3,3, 1,1,1, 1,1,1)(input)
    local enc1 = cudnn.ReLU(true)(enc1)
    local base = base_net(enc1)
    local out = cudnn.VolumetricConvolution( 16,channels_out, 3,3,3, 1,1,1, 1,1,1)(base)
    
    local net  = nn.gModule({input}, {out})
    -- common.net_he_init(net)
    return net
  else
    local n_grids = 0
    if step == 3 then
      n_grids = 64
      -- n_grids = 13824
    end

    local input = nn.Identity()()

    local enc1 = oc.OctreeConvolutionMM( channels_in+1,16, n_grids)(input)
    local enc1 = oc.OctreeReLU(true)(enc1)
    local base = base_net(enc1) 
    local out = oc.OctreeConvolutionMM( 16,channels_out, n_grids)(base)
    
    local out = oc.OctreeToCDHW()(out)

    local net  = nn.gModule({input}, {out})
    -- common.net_he_init(net)
    return net
  end
end
