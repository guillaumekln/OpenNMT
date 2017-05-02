local Trainer = torch.class('Trainer')

local options = {
  {
    '-save_every', 5000,
    [[Save intermediate models every this many iterations within an epoch.
      If = 0, will not save intermediate models.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-report_every', 50,
    [[Report progress every this many iterations within an epoch.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-async_parallel', false,
    [[When training on multiple GPUs, update parameters asynchronously.]]
  },
  {
    '-async_parallel_minbatch', 1000,
    [[In asynchronous training, minimal number of sequential batches before being parallel.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt()
    }
  },
  {
    '-start_iteration', 1,
    [[If loading from a checkpoint, the iteration from which to start.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-start_epoch', 1,
    [[If loading from a checkpoint, the epoch from which to start.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-end_epoch', 13,
    [[The final epoch of the training. If = 0, train forever unless another stopping condition
      is met (e.g. `-min_learning_rate` is reached).]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt()
    }
  },
  {
    '-curriculum', 0,
    [[For this many epochs, order the minibatches based on source length (from smaller to longer).
      Sometimes setting this to 1 will increase convergence speed.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      train_state = true
    }
  }
}

function Trainer.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Trainer')
  onmt.train.Optim.declareOpts(cmd)
  onmt.train.Saver.declareOpts(cmd)
end

function Trainer:__init(args, model, dicts, firstBatch)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
  self.args.profiler = args.profiler
  self.args.disable_mem_optimization = args.disable_mem_optimization

  self.optim = onmt.train.Optim.new(args)
  self.saver = onmt.train.Saver.new(args, model, self.optim, dicts)

  model:training()

  self.model = model

  if not onmt.train.Saver.checkpointDefined(args) then
    self.params, self.gradParams = model:initParams()
  else
    self.params, self.gradParams = model:getParams()
  end

  -- If enabled, share internal buffers to optimize for memory.
  if not self.args.disable_mem_optimization then
    if not firstBatch then
      _G.logger:error('A first batch is needed to optimize the computation graph for memory')
    else
      onmt.utils.Memory.optimize(model, onmt.utils.Cuda.convert(firstBatch))
    end
  end

  -- Add profiling hooks.
  if self.args.profiler then
    model:enableProfiling()
  end
end

function Trainer:eval(data)
  local loss = 0
  local totalWords = 0

  self.model:evaluate()

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    loss = loss + self.model:forwardComputeLoss(batch)
    totalWords = totalWords + self.model:getOutputLabelsCount(batch)
  end

  self.model:training()

  return math.exp(loss / totalWords)
end

function Trainer:countIterations(data)
  return data:batchCount()
end

function Trainer:trainLoop(...)
  error('unimplemented abstract method')
end

function Trainer:trainEpoch(data, epoch, startIteration, batchOrder)
  startIteration = startIteration or 1

  if not batchOrder then
    batchOrder = torch.linspace(1, data:batchCount(), data.batchCount)
  end

  local epochState = onmt.train.EpochState.new(epoch,
                                               startIteration,
                                               self:countIterations(data),
                                               self.optim:getLearningRate())
  local epochProfiler = onmt.utils.Profiler.new(self.args.profiler)

  epochProfiler:add('train')
  self:trainLoop(data, batchOrder, epochState, epochProfiler)
  epochProfiler:stop('train')

  if self.args.profiler then
    _G.logger:info('profile: %s', epochProfiler:log())
  end

  return epochState
end

function Trainer:train(trainData, validData, trainStates)
  local batchOrder

  -- Restore previous training states if defined.
  if trainStates then
    if trainStates.rngStates then
      onmt.utils.Cuda.setRNGStates(trainStates.rngStates)
    end
    if trainStates.optimStates then
      self.optim:setOptimStates(trainStates.optimStates)
    end
    if trainStates.batchOrder and self.args.start_epoch > self.args.curriculum then
      batchOrder = trainStates.batchOrder
    end
  end

  local startEpoch = self.args.start_epoch
  local endEpoch

  if self.args.end_epoch > 0 then
    endEpoch = self.args.end_epoch
    _G.logger:info('Start training from epoch %d to %d...', startEpoch, endEpoch)
  else
    endEpoch = math.huge
    _G.logger:info('Start training from epoch %d and indefinitely...', startEpoch)
  end

  for epoch = startEpoch, endEpoch do
    _G.logger:info('')

    if trainData.sample then
      trainData:sample()
    end

    -- Shuffle batch order past the -curriculum first epochs.
    if not batchOrder and epoch > self.args.curriculum then
      batchOrder = torch.randperm(trainData:batchCount())
    end

    local epochState = self:trainEpoch(trainData, epoch, self.args.start_iteration, batchOrder)
    local validPpl = self:eval(validData)

    _G.logger:info('Validation perplexity: %.2f', validPpl)

    self.optim:updateLearningRate(validPpl, epoch)
    self.saver:saveEpoch(validPpl, epochState)

    -- Early stopping?
    if self.optim:isFinished() then
      _G.logger:warning('Stopping training due to a too small learning rate value.')
      break
    end

    -- Reset batch ordering for the next epoch.
    batchOrder = nil
    self.args.start_iteration = 1
  end
end

return Trainer
