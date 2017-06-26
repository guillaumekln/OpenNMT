local tds = require('tds')
local threads = require('threads')

--[[ A ParallelDataset reads data from one or multiple sources. ]]
local ParallelDataset = torch.class('ParallelDataset')

--[[ Creates a new ParallelDataset. ]]
function ParallelDataset:__init(fileIterators)
  self:_setIterators(fileIterators)
  self.batchSize = 1
  self.discarded = 0
  self.mutex = threads.Mutex()
end

--[[ Returns next batch of entries.

Note: this method is thread-safe.

]]
function ParallelDataset:getNext()
  local entries = tds.Vec()

  self.mutex:lock()

  for _ = 1, self.batchSize do
    local entry = self:_readNext()
    if entry then
      entries:insert(entry)
    else
      break
    end
  end

  self.mutex:unlock()

  if #entries > 0 then
    return entries
  else
    return nil
  end
end

--[[ Return up to `batchSize` entries. ]]
function ParallelDataset:batchify(batchSize)
  self.batchSize = batchSize
  return self
end

--[[ Apply `funcs` on each read items. ]]
function ParallelDataset:map(funcs)
  self.mapFuncs = funcs
  return self
end

--[[ Call `funcs` on each read items (after mapping functions). ]]
function ParallelDataset:iter(funcs)
  self.iterFuncs = funcs
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

--[[ Validates an `entry` against validators. ]]
function ParallelDataset:_isValid(entry)
  if self.parallelValidator then
    if not self.parallelValidator(entry) then
      self.discarded = self.discarded + 1
      return false
    end
  end

  if self.validators then
    for i = 1, #entry do
      if not self.validators[i](entry[i]) then
        self.discarded = self.discarded + 1
        return false
      end
    end
  end

  return true
end

--[[ Set file iterators. ]]
function ParallelDataset:_setIterators(fileIterators)
  self.fileIterators = fileIterators

  self.maps = {}

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
        if self.mapFuncs and self.mapFuncs[i] then
          item = self.mapFuncs[i](item)
        end
        if self.iterFuncs and self.iterFuncs[i] then
          self.iterFuncs[i](item)
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

--[[

  Tries to retrieve a complete entry for the identifier `identifier`. If found, it is removed
  from the temporary storage and returned, otherwise `nil` is returned.

]]
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
