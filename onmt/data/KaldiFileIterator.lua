local String = require('onmt.utils.String')

--[[ FileIterator that consumes Kaldi's .ark format:

KEY [
FEAT1.1 FEAT1.2 FEAT1.3 ... FEAT1.n
...
FEATm.1 FEATm.2 FEATm.3 ... FEATm.n ]

This format is expected to be indexed.

]]
local KaldiFileIterator, parent = torch.class('KaldiFileIterator', 'FileIterator')

function KaldiFileIterator:__init(filename, transformer)
  parent.__init(self, filename, transformer, true)
end

--[[ Read the next item as a `FloatTensor`. ]]
function KaldiFileIterator:_read()
  local values = {}
  local completed = false

  while not completed do
    local line = self.file:read()

    if not line then
      error('unexpected end of file')
    end

    local parts = String.split(String.strip(line), ' ')

    if parts[1] == '[' then
      table.remove(parts, 1)
    end

    if parts[#parts] == ']' then
      table.remove(parts)
      completed = true
    end

    local row = {}
    for i = 1, #parts do
      table.insert(row, tonumber(parts[i]))
    end
    if #row > 0 then
      table.insert(values, row)
    end
  end

  return torch.FloatTensor(values)
end

return KaldiFileIterator
