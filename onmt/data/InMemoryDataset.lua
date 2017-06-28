require('onmt.data.ParallelDataset')

local tds = require('tds')

--[[

  InMemoryDataset is a dataset that stores all entries at once and allow document-based
  operations (e.g. shuffle, sorting, sampling, etc.).

]]
local InMemoryDataset, parent = torch.class('InMemoryDataset', 'ParallelDataset')

--[[ Creates a new InMemoryDataset. ]]
function InMemoryDataset:__init()
  self.data = tds.Vec()
  self.discarded = 0
end

--[[ Apply `funcs` on stored and future items. ]]
function InMemoryDataset:map(funcs)
  parent.map(self, funcs)

  for i = 1, #self.data do
    for j = 1, #funcs do
      if funcs[j] then
        self.data[i][j] = funcs[j](self.data[i][j])
      end
    end
  end

  return self
end

--[[ Call `funcs` on stored and future items (if the latter, it will be called after mapping functions. ]]
function InMemoryDataset:iter(funcs)
  parent.iter(self, funcs)

  for i = 1, #self.data do
    for j = 1, #funcs do
      if funcs[j] then
        funcs[j](self.data[i][j])
      end
    end
  end

  return self
end

function InMemoryDataset:getNext()
  error('InMemoryDataset do not support streaming interface')
end

--[[ Get batch at index `index`. ]]
function InMemoryDataset:getBatch(index)
  local offset
  local size

  if self.batchRange then
    offset = self.batchRange[index].offset
    size = self.batchRange[index].size
  else
    offset = index
    size = 1
  end

  local entries = tds.Vec()

  for i = offset, offset + size - 1 do
    entries:insert(self.data[i])
  end

  return entries
end

--[[ Fills dataset from data files. ]]
function InMemoryDataset:fromFiles(fileIterators, reportEvery)
  self:_setIterators(fileIterators)

  local count = 0

  while true do
    local entry = self:_readNext(fileIterators)

    if not entry then
      break
    elseif self:_isValid(entry) then
      self.data:insert(entry)
    end

    count = count + 1
    if reportEvery and count % reportEvery == 0 then
      _G.logger:info('... %d entries prepared', count)
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

--[[ Prepares batches with up to `batchSize` entries.

`stepFunc` is function that take the current and previous item and returns true
if a batch needs to be terminated.

 ]]
function InMemoryDataset:batchify(batchSize, stepFunc)
  self.batchRange = {}

  local currentBatchSize = 0
  local cursor = 1

  for i = 1, self:size() do
    if currentBatchSize == batchSize
    or stepFunc and i > 1 and stepFunc(self.data[i], self.data[i - 1]) then
      table.insert(self.batchRange, { offset = cursor, size = currentBatchSize })
      cursor = i
      currentBatchSize = 1
    else
      currentBatchSize = currentBatchSize + 1
    end
  end

  table.insert(self.batchRange, { offset = cursor, size = currentBatchSize })

  return self
end

--[[ Returns the number of items in the dataset. ]]
function InMemoryDataset:size()
  return #self.data
end

--[[ Returns the number of batches. ]]
function InMemoryDataset:batchCount()
  if self.batchRange then
    return #self.batchRange
  else
    return self:size()
  end
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

  local newData = tds.Vec()
  newData:resize(#self.data)

  for i = 1, #self.data do
    newData[i] = self.data[perm[i]]
  end

  self.data = newData

  return self
end

return InMemoryDataset
