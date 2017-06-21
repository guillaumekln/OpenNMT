local VectorPreprocessor, parent = torch.class('VectorPreprocessor', 'Preprocessor')

function VectorPreprocessor:__init(maxLength)
  parent.__init(self, maxLength)
end

function VectorPreprocessor:next(file)
  local values = {}
  local completed = false

  while not completed do
    local line = file:read()
    local parts = onmt.utils.String.split(line, ' ')

    if parts[1] == '[' then
      table.remove(parts, 1)
    elseif parts[#parts] == ']' then
      table.remove(parts)
      completed = true
    end

    local row = {}
    for i = 1, #parts do
      table.insert(row, tonumber(parts[i]))
    end
    table.insert(values, row)
  end

  local tensor = torch.FloatTensor(values)
  return self:process(tensor)
end

function VectorPreprocessor:process(tensor)
  local data = {}
  data.input = tensor
  data.length = tensor:size(1)
  return data
end

return VectorPreprocessor
