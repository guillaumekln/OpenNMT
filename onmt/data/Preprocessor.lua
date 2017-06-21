function Preprocessor:__init(maxLength)
  self.maxLength = maxLength
end

function Preprocessor:isValid(data)
  return data.length > 0 and (not self.maxLength or data.length <= self.maxLength)
end
