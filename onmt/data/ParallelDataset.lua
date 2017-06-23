local tds = require('tds')
local Table = require('onmt.utils.Table')

--[[ A ParallelDataset stores data from one or multiple sources. ]]
local ParallelDataset = torch.class('ParallelDataset')

--[[ Creates a new ParallelDataset.

Parameters:

  * `data` - pre-built data (e.g. from preprocessing)

]]
function ParallelDataset:__init(data)
  self.data = data or tds.Vec()
end

--[[ Sets items validators. Entries with one items that is not validated are ignored. ]]
function ParallelDataset:setValidators(validators)
  self.validators = validators
  return self
end

--[[ Sets parallel items validators (e.g. they must be of the same length). ]]
function ParallelDataset:setParallelValidator(validator)
  self.parallelValidator = validator
  return self
end

--[[ Validate an `entry` against validators. ]]
function ParallelDataset:isValid(entry)
  if self.parallelValidator then
    if not self.parallelValidator(entry) then
      return false
    end
  end

  if self.validators then
    for i = 1, #entry do
      if not self.validators[i](entry[i]) then
        return false
      end
    end
  end

  return true
end

--[[ Inserts one entry in the dataset.

Parameters:

  * `entry` - the entry to insert.

Returns: the number of entries inserted.

]]
function ParallelDataset:insert(entry)
  if not self:isValid(entry) then
    return 0
  end

  for i = 1, #entry do
    if torch.isTensor(entry[i]) then
      if #self.data[i] == 0 then
        self.data[i]:insert(tds.Vec())
      end
      self.data[i][1]:insert(entry[i])
    else
      local numStreams = #entry[i]
      if #self.data[i] ~= numStreams then
        for _ = 1, numStreams do
          self.data[i]:insert(tds.Vec())
        end
      end
      for j = 1, numStreams do
        self.data[i][j]:insert(entry[i][j])
      end
    end
  end

  return 1
end

--[[ Fills dataset from data files. ]]
function ParallelDataset:fill(fileIterators, reportEvery)
  local count = 0

  local maps = {}

  for i = 1, #fileIterators do
    self.data[i] = tds.Vec()
    table.insert(maps, tds.Hash())
  end

  while not fileIterators[1]:isEOF() do
    local items = {}
    local ids = {}

    -- Retrieve next items and ids from all files.
    for i = 1, #fileIterators do
      local item, id = fileIterators[i]:next()
      items[i] = item
      ids[i] = id
    end

    -- Check if all items refer to same entry.
    local aligned = true
    for i = 2, #ids do
      if ids[i] ~= ids[1] then
        aligned = false
      end
    end

    if aligned then
      self:insert(items)
    else
      -- If items are not aligned, save the id -> item mapping.
      for i = 1, #items do
        if items[i] then
          if maps[i][ids[i]] then
            error('duplicate item identifier: ' .. tostring(ids[i]))
          else
            maps[i][ids[i]] = items[i]
          end
        end
      end
    end

    count = count + 1
    if reportEvery and count % reportEvery == 0 then
      _G.logger:info('... ' .. count .. ' entries processed')
    end
  end

  -- Process remaining unaligned items.
  for id, item in pairs(maps[1]) do
    local items = { item }
    for i = 2, #fileIterators do
      local item2 = maps[i][id]
      if not item2 then
        error('missing item for identifier ' .. tostring(id))
      else
        table.insert(items, item2)
      end
    end
    self:insert(items)
  end

  return self
end

--[[ Returns the number of streams of the `index`-th file. ]]
function ParallelDataset:getNumStreams(index)
  return #self.data[index]
end

--[[ Returns the number of items in the dataset. ]]
function ParallelDataset:size()
  return #self.data[1][1]
end

--[[ Permutes all entries using `perm` tensor. ]]
function ParallelDataset:permute(perm)
  assert(perm:size(1) == self:size())

  for i = 1, #self.data do
    for j = 1, #self.data[i] do
      self.data[i][j] = Table.reorder(self.data[i][j], perm, true)
    end
  end

  return self
end

--[[ Shuffles items. ]]
function ParallelDataset:shuffle()
  local perm = torch.randperm(self:size())
  return self:permute(perm)
end

--[[ Sorts all entries based on lengths from the `index`-th file. ]]
function ParallelDataset:sortByLength(index)
  local lengths = torch.IntTensor(self:size())

  for i = 1, #self.data[index][1] do
    lengths[i] = self.data[index][1][i]:size(1)
  end

  local _, perm = torch.sort(lengths)
  return self:permute(perm)
end

return ParallelDataset
