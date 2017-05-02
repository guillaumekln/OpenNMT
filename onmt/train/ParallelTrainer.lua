local threads = require('threads')

local ParallelTrainer, parent = torch.class('ParallelTrainer', 'Trainer')

-- Synchronizes the current stream on dst device with src device. This is only
-- necessary if we are not on the default stream
function ParallelTrainer.waitForDevice(dst, src)
  local stream = cutorch.getStream()
  if stream ~= 0 then
    cutorch.streamWaitForMultiDevice(dst, stream, { [src] = {stream} })
  end
end

function ParallelTrainer:__init(opt, instances, ...)
  parent.__init(self, opt, ...)

  self.params = { self.params }
  self.gradParams = { self.gradParams }

  self.instances = math.max(instances, 1)

  if self.instances > 1 then
    self.gradBuffer = onmt.utils.Cuda.convert(torch.Tensor())

    local globalLogger = _G.logger
    local globalProfiler = _G.profiler

    threads.Threads.serialization('threads.sharedserialize')

    self._pool = threads.Threads(
      self.instances,
      function()
        -- require('cunn')
        -- require('nngraph')
        -- require('onmt.init')
      end,
      function(threadid)
        _G.logger = globalLogger
        _G.profiler = globalProfiler
        --onmt.utils.Cuda.init(opt, threadid)
      end
    ) -- dedicate threads to GPUs

    self._pool:specific(true)
  end

  -- Create network replicas.
  self:dispatch(function(idx)
    _G.model = idx == 1 and self.model or onmt.utils.Tensor.deepClone(self.model)

    if self.params[idx] then
      _G.params, _G.gradParams = self.params[idx], self.gradParams[idx]
    else
      _G.params, _G.gradParams = _G.model:getParams()
    end

    return idx, _G.params, _G.gradParams
  end, function(idx, params, gradParams)
    self.params[idx] = params
    self.gradParams[idx] = gradParams
  end)
end

function ParallelTrainer:dispatch(func, callback)
  callback = callback or function() end

  for j = 1, self.instances do
    if not self._pool then
      callback(func(j))
    else
      self._pool:addjob(j, function() return func(j) end, callback)
    end
  end

  if self._pool then
    self._pool:synchronize()
  end
end

return ParallelTrainer
