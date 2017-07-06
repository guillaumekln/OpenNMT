local tds = require('tds')
local threads = require('threads')

--[[ A `ParallelDataset` reads indexed entries from one or multiple sources. ]]
local ParallelDataset = torch.class('ParallelDataset')

--[[ Creates a new `ParallelDataset`.

Parameters:

  * `fileIterators` - a table of `FileIterator`.
  * `bufferSize` - the number of entries to read at once. If = 0, bufferize the whole dataset.

]]
function ParallelDataset:__init(fileIterators, bufferSize)
  self.fileIterators = fileIterators

  -- Maps are used to align indexed entries.
  self.maps = {}
  for i = 1, #fileIterators do
    self.maps[i] = tds.Hash()
  end

  self.remainingLoops = 1
  self.batchSize = 1
  self.bufferSize = bufferSize or 1
  self.buffer = tds.Vec()
  self.mutex = threads.Mutex()
end

--[[ Returns an iterator on the dataset. ]]
function ParallelDataset:iterate()
  return function()
    return self:getNext()
  end
end

--[[ Returns next batch of entries.

Note: this method is thread-safe.

]]
function ParallelDataset:getNext()
  local entries = tds.Vec()

  self.mutex:lock()

  while #entries < self.batchSize do
    if #self.buffer == 0 then
      self:_bufferize()
    end

    -- No more entries to read.
    if #self.buffer == 0 then
      if self.remainingLoops then
        self.remainingLoops = self.remainingLoops - 1
      end

      -- Reset iterators and continue if defined.
      if not self.remainingLoops or self.remainingLoops > 0 then
        for i = 1, #self.fileIterators do
          self.fileIterators[i]:reset()
        end

        self:_bufferize()
      else
        break
      end
    end

    if #entries == 0
    or self.batchStepFunc and self.batchStepFunc(entries[#entries], self.buffer[#self.buffer]) then
      entries:insert(self.buffer[#self.buffer])
      self.buffer:remove()
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

--[[ Returns `count` batches. ]]
function ParallelDataset:take(count)
  local batches = tds.Vec()

  while #batches < count do
    local batch = self:getNext()

    if not batch then
      break
    else
      batches:insert(batch)
    end
  end

  return batches
end

--[[ Will loop `loops` times over the dataset. If nil, will loop indefinitely. ]]
function ParallelDataset:loop(loops)
  self.remainingLoops = loops
  return self
end

--[[ Will return up to `batchSize` entries.

Parameters:

  * `batchSize` - the maximum batch size.
  * `stepFunc` - an optional callable that takes two consecutive entries and returns false
    if the current batch needs to stop.

]]
function ParallelDataset:batchify(batchSize, stepFunc)
  self.batchSize = batchSize
  self.batchStepFunc = stepFunc
  return self
end

--[[ Will shuffle entries. ]]
function ParallelDataset:shuffle()
  self.shuffle = true
  return self
end

--[[ Will sort entries by key values.

Parameters:

  * `sortKeyFunc` - a callable that returns a key per entry.

]]
function ParallelDataset:sort(sortKeyFunc)
  self.sortKeyFunc = sortKeyFunc
  return self
end

--[[ Will apply `funcs` on each read items. ]]
function ParallelDataset:map(funcs)
  self.mapFuncs = funcs
  return self
end

--[[ Will call `funcs` on each read items (after mapping functions). ]]
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

--[[ Permutes buffered entries using `perm` tensor. ]]
function ParallelDataset:_permute(perm)
  assert(perm:size(1) == #self.buffer)

  local newBuffer = tds.Vec()
  newBuffer:resize(#self.buffer)

  for i = 1, #self.buffer do
    newBuffer[i] = self.buffer[perm[i]]
  end

  self.buffer = newBuffer

  return self
end

--[[ Bufferizes entries. ]]
function ParallelDataset:_bufferize()
  while self.bufferSize == 0 or #self.buffer < self.bufferSize do
    local entry = self:_readNext()

    if not entry then
      break
    end

    if self:_isValid(entry) then
      self.buffer:insert(entry)
    end
  end

  -- Also shuffle and sort if defined.
  if #self.buffer > 0 then
    if self.shuffle then
      local perm = torch.randperm(#self.buffer)
      self:_permute(perm)
    end

    if self.sortKeyFunc then
      local values = torch.IntTensor(#self.buffer)

      for i = 1, values:size(1) do
        -- Inverse key value as buffered entries will be consumed in reverse order (pop operation).
        values[i] = -self.sortKeyFunc(self.buffer[i])
      end

      local _, perm = torch.sort(values)
      self:_permute(perm)
    end
  end

  return self
end

--[[ Reads next entry from files. ]]
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
        -- Apply registered functions on read item.
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
