local Processor = torch.class('Processor')

--[[ Base class for preprocessors: classes that process raw inputs into tensors. ]]
function Processor:__init(maxLength)
  self.maxLength = maxLength
end

--[[ Consume next item from a file and process it.

Parameters:

  * `file` - a file handle.
  * `func` - a callable to process the item.

Returns: the result of `func` applied to the next item, or nil if EOF is reached.

]]
function Processor:next(file, func)
  local item = self:consume(file)

  if not item then
    return nil
  else
    return func(self, item)
  end
end

--[[ Consume the next item from a file.

Parameters:

  * `file` - a file handle.

Returns: the next item.

]]
function Processor:consume(_)
  error('Not implemented')
end

-- Process single item.
function Processor:process(_)
  error('Not implemented')
end

function Processor:isValid(data)
  return data.length > 0 and (not self.maxLength or data.length <= self.maxLength)
end

return Processor
