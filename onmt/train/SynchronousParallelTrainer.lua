local nccl

local SynchronousParallelTrainer, parent = torch.class('SynchronousParallelTrainer', 'ParallelTrainer')

function SynchronousParallelTrainer:__init(opt, ...)
  parent.__init(self, opt, ...)

  if not opt.no_nccl then
    -- check if we have nccl installed
    local ret
    ret, nccl = pcall(require, 'nccl')
    if not ret then
      _G.logger:warning("For improved efficiency with multiple GPUs, consider installing nccl")
      nccl = nil
    elseif os.getenv('CUDA_LAUNCH_BLOCKING') == '1' then
      _G.logger:warning("CUDA_LAUNCH_BLOCKING enabled: cannot use nccl")
      nccl = nil
    end
  end
end

function SynchronousParallelTrainer:countIterations(data)
  -- In parallel mode, the number of iterations is reduced to reflect larger batch size.
  return math.ceil(data:batchCount() / self.instances)
end

function SynchronousParallelTrainer:trainLoop(data, batchOrder, epochState, epochProfiler)
  local startIteration = epochState.iterations + 1
  local epoch = epochState.epoch
  local needLog = false
  local iter = startIteration

  -- Keep local references of attributes used in threads.
  local optim = self.optim
  local doProfile = self.args.profiler

  for i = startIteration, data:batchCount(), self.instances do
    local batches = {}
    local totalSize = 0
    needLog = true

    for j = 1, math.min(self.instances, data:batchCount() - i + 1) do
      local batchIdx = batchOrder[i + j - 1]
      table.insert(batches, data:getBatch(batchIdx))
      totalSize = totalSize + batches[#batches].size
    end

    local losses = {}
    local indvAvgLosses = {}

    self:dispatch(function(idx)
      _G.profiler = onmt.utils.Profiler.new(doProfile)

      _G.batch = batches[idx]
      if _G.batch == nil then
        return idx, 0, nil, _G.profiler:dump()
      end

      -- Send batch data to the GPU.
      onmt.utils.Cuda.convert(_G.batch)
      _G.batch.totalSize = totalSize

      optim:zeroGrad(_G.gradParams)
      local loss, indvAvgLoss = _G.model:trainNetwork(_G.batch)

      return idx, loss, indvAvgLoss, _G.profiler:dump()
    end,
    function(idx, loss, indvAvgLoss, profile)
      losses[idx] = loss
      if data.needIndividualLosses and data:needIndividualLosses() then
        indvAvgLosses[idx] = indvAvgLoss
      end
      epochProfiler:add(profile)
    end)

    if self.instances > 1 then
      -- Accumulate the gradients from the different parallel threads.
      SynchronousParallelTrainer.accGradParams(self.gradParams, batches)
    end

    -- Update the parameters.
    optim:prepareGrad(self.gradParams[1])
    optim:updateParams(self.params[1], self.gradParams[1])

    if self.instances > 1 then
      -- Synchronize the parameters with the different parallel threads.
      SynchronousParallelTrainer.syncParams(self.params)
    end

    for bi = 1, #batches do
      epochState:update(self.model, batches[bi], losses[bi])
      if data.needIndividualLosses and data:needIndividualLosses() then
        data:setLoss(batchOrder[i + bi - 1], indvAvgLosses[bi])
      end
    end

    if iter % self.args.report_every == 0 then
      epochState:log(iter)
      needLog = false
    end
    if self.args.save_every > 0 and iter % self.args.save_every == 0 then
      self.saver:saveIteration(iter, epochState, batchOrder)
    end
    iter = iter + 1
  end

  if needLog then
    epochState:log(iter)
  end
end

--[[ Accumulate the gradient parameters from the different parallel threads. ]]
function SynchronousParallelTrainer.accGradParams(gradParams, batches)
  for h = 1, #gradParams[1] do
    local inputs = { gradParams[1][h] }
    for j = 2, #batches do
      if not nccl then
        -- TODO - this is memory costly since we need to clone full parameters from one GPU to another
        -- to avoid out-of-memory, we can copy/add by batch

        -- Synchronize before and after copy to ensure that it doesn't overlap
        -- with this add or previous adds
        ParallelTrainer.waitForDevice(onmt.utils.Cuda.gpuIds[j], onmt.utils.Cuda.gpuIds[1])
        local remoteGrads = onmt.utils.Tensor.reuseTensor(self.gradBuffer, gradParams[j][h]:size())
        remoteGrads:copy(gradParams[j][h])
        ParallelTrainer.waitForDevice(onmt.utils.Cuda.gpuIds[1], onmt.utils.Cuda.gpuIds[j])
        gradParams[1][h]:add(remoteGrads)
      else
        table.insert(inputs, gradParams[j][h])
      end
    end
    if nccl then
      nccl.reduce(inputs, nil, true)
    end
  end
end

--[[ Sync parameters from main model to different parallel threads. ]]
function SynchronousParallelTrainer.syncParams(params)
  if not nccl then
    for j = 2, #params do
      for h = 1, #params[1] do
        params[j][h]:copy(params[1][h])
      end
      ParallelTrainer.waitForDevice(onmt.utils.Cuda.gpuIds[j], onmt.utils.Cuda.gpuIds[1])
    end
  else
    for h = 1, #params[1] do
      local inputs = { params[1][h] }
      for j = 2, #params do
        table.insert(inputs, params[j][h])
      end
      nccl.bcast(inputs, true, 1)
    end
  end
end

return SynchronousParallelTrainer
