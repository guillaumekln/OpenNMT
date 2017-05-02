local threads = require('threads')
local tds

local AsynchronousParallelTrainer, parent = torch.class('AsynchronousParallelTrainer', 'ParallelTrainer')

function AsynchronousParallelTrainer:__init(...)
  if not tds then
    tds = require('tds')
  end

  parent.init(self, ...)

  self.mutex = threads.Mutex()
end

function AsynchronousParallelTrainer:trainLoop(data, batchOrder, epochState, epochProfiler)
  local masterGPU = onmt.utils.Cuda.gpuIds[1]
  local counter = tds.AtomicCounter()
  local mutexId = self.mutex:id()
  local startInteration = epochState.iterations + 1
  local iter = 0
  local needLog = false
  local minBatch = self.args.async_parallel_minbatch
  local mainParams = self.params[1]
  local maxConcurrentIter = self.args.report_every
  if self.args.save_every > 0 and self.args.save_every < maxConcurrentIter then
    maxConcurrentIter = self.args.save_every
  end

  -- Keep local references of attributes used in threads.
  local optim = self.optim
  local doProfile = self.args.profiler
  local gradBuffer = self.gradBuffer

  counter:set(startIteration)

  while counter:get() <= data:batchCount() do
    needLog = true
    local startCounter = counter:get()

    self:dispatch(function(idx)
      _G.profiler = onmt.utils.Profiler.new(doProfile)
      -- First GPU is only used for master parameters.
      -- Use 1 GPU only for 1000 first batch.
      if idx == 1 or (idx > 2 and epoch == 1 and counter:get() < minBatch) then
        return
      end

      local batches = {}
      local losses = {}
      local indvAvgLosses = {}

      while true do
        local i = counter:inc()
        if i - startCounter >= maxConcurrentIter or i > data:batchCount() then
          return batches, losses, indvAvgLosses, _G.profiler:dump()
        end

        local batchIdx = getBatchIdx(i)

        _G.batch = data:getBatch(batchIdx)
        table.insert(batches, onmt.utils.Tensor.deepClone(_G.batch))
        onmt.utils.Cuda.convert(_G.batch)

        optim:zeroGrad(_G.gradParams)
        local loss, indvAvgLoss = _G.model:trainNetwork(_G.batch)
        table.insert(losses, loss)
        if data.needIndividualLosses and data:needIndividualLosses() then
          indvAvgLosses[batchIdx] = indvAvgLoss
        end

        -- Update the parameters.
        optim:prepareGrad(_G.gradParams)

        -- Add up gradParams to params and synchronize back to this thread.
        onmt.utils.Parallel.updateAndSync(mainParams, _G.gradParams, _G.params, gradBuffer, masterGPU, mutexId)
      end
    end,
    function(batches, losses, indvAvgLosses, profile)
      if batches then
        iter = iter + #batches
        for i = 1, #batches do
          epochState:update(self.model, batches[i], losses[i])
          if data.needIndividualLosses and data:needIndividualLosses() then
            data:setLoss(getBatchIdx(i), indvAvgLosses[getBatchIdx(i)])
          end
        end
        epochProfiler:add(profile)
      end
    end)

    if iter % self.args.report_every == 0 then
      epochState:log()
      needLog = false
    end
    if iter % self.args.save_every == 0 then
      self.saver:saveIteration(iter, epochState, batchOrder)
    end
  end

  if needLog then
    epochState:log(iter)
  end
end

-- [[ In async mode, sync the parameters from all replica to master replica. ]]
function AsynchronousParallelTrainer.updateAndSync(masterParams, replicaGradParams, replicaParams, gradBuffer, masterGPU, gmutexId)
  -- Add a mutex to avoid competition while accessing shared buffer and while updating parameters.
  local mutex = threads.Mutex(gmutexId)
  mutex:lock()
  local device = cutorch.getDevice()
  cutorch.setDevice(masterGPU)
  for h = 1, #replicaGradParams do
    waitForDevice(device, masterGPU)
    local remoteGrads = onmt.utils.Tensor.reuseTensor(gradBuffer, replicaGradParams[h]:size())
    remoteGrads:copy(replicaGradParams[h])
    waitForDevice(masterGPU, device)
    masterParams[h]:add(remoteGrads)
  end
  cutorch.setDevice(device)
  for h = 1, #replicaGradParams do
    replicaParams[h]:copy(masterParams[h])
    waitForDevice(device, masterGPU)
  end
  mutex:unlock()
end

return AsynchronousParallelTrainer
