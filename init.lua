require 'cutorch'
require 'nn'

prnn = {}

require('prnn.ffi')
local C = prnn.C
local ffi = require 'ffi'

local maxStreamsPerDevice = 1024
local numDevices = cutorch.getDeviceCount()
-- this tensor keeps track of whether a handle has been initialized or not
local handleStatus = torch.ByteTensor(numDevices,
                                  maxStreamsPerDevice):zero()
-- here we create an array of prnn handle structs
prnn.handle = ffi.new('struct prnnContext*[?]', numDevices*maxStreamsPerDevice)
local function destroy(handle)
    local currentDevice = cutorch.getDevice()
    for i=1,numDevices do
        cutorch.setDevice(i)
        -- streams go from 0 to maxStreamsPerDevice - 1
        for j=0,maxStreamsPerDevice - 1 do
            if handleStatus[i][j + 1] == 1 then -- if handle was created
                errcheck('prnnDestroy', handle[(((i-1)*maxStreamsPerDevice) + j)]);
            end
        end
    end
    cutorch.setDevice(currentDevice)
end
ffi.gc(prnn.handle, destroy)

prnn.typemap = {
   ['torch.CudaHalfTensor']   = 'PRNN_DATA_HALF',
   ['torch.CudaTensor']       = 'PRNN_DATA_FLOAT',
   ['torch.CudaDoubleTensor'] = 'PRNN_DATA_DOUBLE',
}

-- TODO: determine if device supports true half and use true half on it
-- so far use float for half and float, double for double
local function determineHalfCapability(dev)
   local prop = cutorch.getDeviceProperties(dev)
   if prop.major >= 6 or prop.name:find'X1' then
      return 'PRNN_DATA_HALF'
   else
      return 'PRNN_DATA_FLOAT'
   end
end

local configmaps = {}
for i=1,cutorch.getDeviceCount() do
   configmaps[i] = {
      ['torch.CudaHalfTensor']   = determineHalfCapability(i),
      ['torch.CudaTensor']       = 'PRNN_DATA_FLOAT',
      ['torch.CudaDoubleTensor'] = 'PRNN_DATA_DOUBLE',
   }
end

prnn.configmap = function(tensortype)
   return configmaps[cutorch.getDevice()][tensortype]
end

function prnn.getHandle()
    local device = cutorch.getDevice()
    local stream = cutorch.getStream() -- starts from 0
    assert(stream < maxStreamsPerDevice, 'prnn bindings only support max of : '
               .. maxStreamsPerDevice .. ' streams per device')
    -- lazy initialization of handles
    if handleStatus[device][stream + 1] == 0 then
        local status = C['prnnCreate'](prnn.handle
                                        + (((device-1) * maxStreamsPerDevice)
                                                + stream))
        if status ~= ffi.C.PRNN_STATUS_SUCCESS then
            local str = ffi.string(C.prnnGetErrorString(status))
            error('Error in PRNN: ' .. str)
        end
        handleStatus[device][stream + 1] = 1 -- mark handle as initialized
    end
    return prnn.handle[(((device-1)*maxStreamsPerDevice) + stream)]
end

local errcheck = function(f, ...)
    C.prnnSetStream(prnn.getHandle(),
                     ffi.C.THCState_getCurrentStream(cutorch.getState()))
   local status = C[f](...)
   if status ~= ffi.C.PRNN_STATUS_SUCCESS then
      local str = ffi.string(C.prnnGetErrorString(status))
      error('Error in PRNN: ' .. str .. ' ('..f..')')
   end
end
prnn.errcheck = errcheck

function prnn.toDescriptor(t)
   local typename = torch.typename(t)
   assert(prnn.typemap[typename])
   local descriptor = ffi.new('struct prnnTensorStruct*[1]')
   -- create descriptor
   errcheck('prnnCreateTensorDescriptor', descriptor)
   -- set gc hook
   local function destroy(d)
      errcheck('prnnDestroyTensorDescriptor', d[0]);
   end
   ffi.gc(descriptor, destroy)
   -- view 2D and 3D as 4D
   if t:dim() == 2 then
      t = t:view(t:size(1), t:size(2), 1, 1)
   elseif t:dim() == 3 then
      t = t:view(t:size(1), t:size(2), t:size(3), 1)
   end
   -- set descriptor
   local size = torch.LongTensor(t:size()):int()
   local stride = torch.LongTensor(t:stride()):int()

   errcheck('prnnSetTensorNdDescriptor', descriptor[0], prnn.typemap[typename],
            t:dim(), size:data(), stride:data())
   return descriptor
end


local sharedBuffer = {}
for i=1,numDevices do
    sharedBuffer[i] = {}
end

function prnn.getSharedWorkspace()
    local device = cutorch.getDevice()
    local stream = cutorch.getStream() -- starts from 0
    if not sharedBuffer[device][stream] then
        sharedBuffer[device][stream] = torch.CudaTensor(1)
    end
    return sharedBuffer[device][stream]
end

require('prnn.RNN')
require('prnn.RNNTanh')
--[[
require('prnn.RNNReLU')
require('prnn.BLSTM')
require('prnn.LSTM')
require('prnn.BGRU')
require('prnn.GRU')
--]]


return prnn
