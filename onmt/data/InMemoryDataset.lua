require('onmt.data.ParallelDataset')

local tds = require('tds')
local Table = require('onmt.utils.Table')

--[[

  InMemoryDataset is a dataset that stores all entries at once and allow document-based
  operations (e.g. shuffle, sorting, sampling, etc.).

]]
local InMemoryDataset, _ = torch.class('InMemoryDataset', 'ParallelDataset')

--[[ Creates a new InMemoryDataset. ]]
function InMemoryDataset:__init()
  self.data = tds.Vec()
  self.offset = 0
end

function InMemoryDataset:getNext()
  error('InMemoryDataset do not support streaming interface')
end

--[[ Fills dataset from data files. ]]
function InMemoryDataset:fromFiles(fileIterators)
  self:_setIterators(fileIterators)

  while true do
    local entry = self:_readNext(fileIterators)

    if not entry then
      break
    elseif self:_isValid(entry) then
      self.data:insert(entry)
    end
  end

  -- Warn about identifiers with missing items.
  local orphans = {}
  for i = 1, #self.maps do
    for id, _ in pairs(self.maps[i]) do
      orphans[id] = true
    end
  end
  for id, _ in pairs(orphans) do
    _G.logger:warning('Not all files contain the identifier %s', tostring(id))
  end

  self.maps = {}

  return self
end


--[[ Returns the number of items in the dataset. ]]
function InMemoryDataset:size()
  return #self.data
end

--[[ Shuffles items. ]]
function InMemoryDataset:shuffle()
  local perm = torch.randperm(self:size())
  return self:_permute(perm)
end

--[[ Sorts all entries based on values returned by `keyFunc`. ]]
function InMemoryDataset:sort(keyFunc)
  local values = torch.IntTensor(self:size())

  for i = 1, values:size(1) do
    values[i] = keyFunc(self.data[i])
  end

  local _, perm = torch.sort(values)
  return self:_permute(perm)
end

--[[ Permutes all entries using `perm` tensor. ]]
function InMemoryDataset:_permute(perm)
  assert(perm:size(1) == self:size())
  self.data = Table.reorder(self.data, perm, true)
  return self
end

return InMemoryDataset
