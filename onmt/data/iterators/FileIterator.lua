--[[ Class that iterates on a file of items. ]]
local FileIterator = torch.class('FileIterator')

--[[ Creates a new `FileIterator`.

Parameters:

  * `filename` - the file to iterate on.
  * `indexed` - whether items are indexed by an identifier.

]]
function FileIterator:__init(filename, indexed)
  self.filename = filename
  self.file = assert(io.open(filename, 'r'))
  self.indexed = indexed
  self.offset = 0
end

--[[ Closes the file handle. ]]
function FileIterator:close()
  self.file:close()
end

--[[ Resets iterator to the beginning of the file. ]]
function FileIterator:reset()
  self.file:seek('set')
  self.offset = 0
end

--[[ Returns true if the end of file is reached. ]]
function FileIterator:isEOF()
  local current = self.file:seek()
  local eos = self.file:read(1) == nil
  self.file:seek('set', current)
  return eos
end

--[[ Consumes and returns the next item.

Returns:

  * `item` - the item.
  * `id` - the item identifier.

]]
function FileIterator:next()
  if self:isEOF() then
    return nil
  end

  local item, id = self:_read()
  self.offset = self.offset + 1
  return item, id
end

--[[ Returns the next item.

Returns:

  * `item` - the item.
  * `id` - the item identifier.

]]
function FileIterator:lookup()
  if self:isEOF() then
    return nil
  end

  local current = self.file:seek()
  local item, id = self:_read()
  self.file:seek('set', current)
  return item, id
end

--[[ Reads the next entry. ]]
function FileIterator:_read()
  local id
  local item

  local _, err = pcall(function ()
    if self.indexed then
      id = self:_readId()
    else
      id = self.offset + 1
    end

    item = self:_readItem()
  end)

  if err then
    error(err .. ' (' .. self.filename .. ':' .. id .. ')')
  end

  return item, id
end

--[[ Reads the next line identifier as a string. ]]
function FileIterator:_readId()
  local id = ''

  while true do
    local c = self.file:read(1)

    if c == nil then
      error('empty line')
    elseif c == ' ' then
      break
    else
      id = id .. c
    end
  end

  return id
end

--[[ Reads the next item.

This method must be overriden by child classes.

]]
function FileIterator:_readItem()
  error('Not implemented')
end

return FileIterator
