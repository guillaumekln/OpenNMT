--[[ Base class for preprocessors: classes that process raw inputs into tensors. ]]
function Preprocessor:__init(maxLength)
  self.maxLength = maxLength
end

--[[ Consume next item from a file and process it.

Parameters:

  * `file` - a file handle.
  * `func` - a callable to process the item.

Returns: the result of `func` applied to the next item, or nil if EOF is reached.

]]
function Preprocessor:next(file, func)
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
function Preprocessor:consume(_)
  error('Not implemented')
end

-- Process single item.
function Preprocessor:process(_)
  error('Not implemented')
end

function Preprocessor:isValid(data)
  return data.length > 0 and (not self.maxLength or data.length <= self.maxLength)
end
