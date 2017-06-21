local VectorPreprocessor, parent = torch.class('VectorPreprocessor', 'Preprocessor')

function VectorPreprocessor:__init(maxLength)
  parent.__init(self, maxLength)
end

function VectorPreprocessor:consume(file)
  local values = {}
  local completed = false

  while not completed do
    local line = file:read()

    if not line and #values == 0 then
      return nil
    end

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

  return torch.FloatTensor(values)
end

function VectorPreprocessor:process(values)
  local data = {}

  if torch.isTensor(values) then
    data.input = values
  else
    data.input = torch.FloatTensor(values)
  end

  data.length = data.input:size(1)

  return data
end

return VectorPreprocessor
