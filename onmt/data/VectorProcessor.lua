local VectorProcessor, parent = torch.class('VectorProcessor', 'Processor')

function VectorProcessor:__init(maxLength)
  parent.__init(self, maxLength)
end

function VectorProcessor:consume(file)
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

function VectorProcessor:process(values)
  local data = {}
  data.input = torch.isTensor(values) and values or torch.FloatTensor(values)
  data.length = data.input:size(1)
  return data
end

return VectorProcessor
