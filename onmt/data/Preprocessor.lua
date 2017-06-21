function Preprocessor:__init(maxLength)
  self.maxLength = maxLength
end

-- Consume next from file and apply func.
function Preprocessor:next(file, func)
  local item = self:consume(file)

  if not item then
    return nil
  else
    return func(self, item)
  end
end

-- Consume next item in file.
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
