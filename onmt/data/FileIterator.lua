--[[ Class that iterates on a file of items. ]]
local FileIterator = torch.class('FileIterator')

--[[ Create a new FileIterator.

Parameters:

  * `filename` - the file to iterate on.
  * `indexed` - whether items are indexed by an identifier.
  * `func` - optional callable to apply on items.

]]
function FileIterator:__init(filename, indexed, func)
  self.filename = filename
  self.file = assert(io.open(filename, 'r'))
  self.indexed = indexed
  self.func = func
  self.offset = 0
end

--[[ Close the file handle. ]]
function FileIterator:close()
  self.file:close()
end

--[[ Return true if the end of file is reached. ]]
function FileIterator:isEOF()
  local current = self.file:seek()
  local eos = self.file:read(1) == nil
  self.file:seek('set', current)
  return eos
end

--[[ Return the next item.

Returns:

  * `item` - the item.
  * `id` - the item identifier.

]]
function FileIterator:next()
  if self:isEOF() then
    return nil
  end

  self.offset = self.offset + 1

  local id
  local item

  local _, err = pcall(function ()
    if self.indexed then
      id = self:_readId()
    else
      id = self.offset
    end

    item = self:_read()
    if self.func then
      item = self.func(item)
    end
  end)

  if err then
    error(err .. ' (' .. self.filename .. ':' .. id .. ')')
  end

  return item, id
end

--[[ Read the next line identifier as a string. ]]
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

--[[ Read the next item.

This method must be overriden by child classes.

]]
function FileIterator:_read()
  error('Not implemented')
end

return FileIterator
