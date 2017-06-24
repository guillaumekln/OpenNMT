local tds = require('tds')
local Table = require('onmt.utils.Table')

--[[ A ParallelDataset stores data from one or multiple sources. ]]
local ParallelDataset = torch.class('ParallelDataset')

--[[ Creates a new ParallelDataset. ]]
function ParallelDataset:__init(data)
  self.data = data or tds.Vec()
  self.maps = {}
end

--[[ Returns the number of items in the dataset. ]]
function ParallelDataset:size()
  return #self.data
end

--[[ Apply `funcs` on each read items. ]]
function ParallelDataset:map(funcs)
  self.funcs = funcs
  return self
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

--[[ Fills dataset from data files. ]]
function ParallelDataset:fromFiles(fileIterators)
  self:_setIterators(fileIterators)

  while true do
    local entry = self:_readNext(fileIterators)

    if not entry then
      break
    elseif self:_isValid(entry) then
      self.data:insert(entry)
    end
  end

  -- Warned about identifiers with missing items.
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

--[[ Shuffles items. ]]
function ParallelDataset:shuffle()
  local perm = torch.randperm(self:size())
  return self:_permute(perm)
end

--[[ Sorts all entries based on values returned by `keyFunc`. ]]
function ParallelDataset:sort(keyFunc)
  local values = torch.IntTensor(self:size())

  for i = 1, values:size(1) do
    values[i] = keyFunc(self.data[i])
  end

  local _, perm = torch.sort(values)
  return self:_permute(perm)
end

--[[ Permutes all entries using `perm` tensor. ]]
function ParallelDataset:_permute(perm)
  assert(perm:size(1) == self:size())
  self.data = Table.reorder(self.data, perm, true)
  return self
end

--[[ Validate an `entry` against validators. ]]
function ParallelDataset:_isValid(entry)
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

--[[ Set file iterators. ]]
function ParallelDataset:_setIterators(fileIterators)
  self.fileIterators = fileIterators

  for i = 1, #fileIterators do
    self.maps[i] = tds.Hash()
  end
end

--[[ Read next entry from files. ]]
function ParallelDataset:_readNext()
  local nextEntry

  -- It will loop until we can return an aligned entry.
  while not nextEntry do
    local items = tds.Vec()
    local ids = tds.Vec()

    -- Retrieve next items and ids from all files.
    for i = 1, #self.fileIterators do
      local item, id = self.fileIterators[i]:next()

      if item then
        if self.funcs and self.funcs[i] then
          item = self.funcs[i](item)
        end

        items[i] = item
        ids[i] = id
      end
    end

    -- If no items were retrieved, we reached EOF for all files.
    if #items == 0 then
      break
    end

    -- Check if all items refers to the same entry.
    local aligned = #ids == #self.fileIterators
    for i = 2, #ids do
      if ids[i] ~= ids[1] then
        aligned = false
      end
    end

    if aligned then
      nextEntry = items
    else
      -- If items are not aligned, save the mapping id -> item.
      for i = 1, #self.fileIterators do
        if items[i] then
          if self.maps[i][ids[i]] then
            error('duplicate item identifier: ' .. tostring(ids[i]))
          else
            self.maps[i][ids[i]] = items[i]
          end
        end
      end

      -- Check if we can now retrieve an entry.
      for i = 1, #self.fileIterators do
        if ids[i] then
          local entry = self:_retrieveEntry(ids[i])
          if entry then
            nextEntry = entry
            break
          end
        end
      end
    end
  end

  -- After reading the files, there may be some unaligned items left.
  if not nextEntry then
    for id, _ in pairs(self.maps[1]) do
      local entry = self:_retrieveEntry(id)
      if entry then
        nextEntry = entry
        break
      end
    end
  end

  return nextEntry
end

function ParallelDataset:_retrieveEntry(identifier)
  local items = tds.Vec()

  -- Retrieve items and return with nil if one is missing.
  for i = 1, #self.maps do
    if not self.maps[i][identifier] then
      return nil
    else
      items[i] = self.maps[i][identifier]
    end
  end

  -- Delete items from maps.
  for i = 1, #self.maps do
    self.maps[i][identifier] = nil
  end

  return items
end

return ParallelDataset
