--[[ Class that iterates on a file of items. ]]
local FileIterator = torch.class('FileIterator')

--[[ Create a new FileIterator.

Parameters:

  * `filename` - the file to iterate on.
  * `indexed` - whether items are indexed by an identifier.

]]
function FileIterator:__init(filename, indexed)
  self.file = assert(io.open(filename, 'r'))
  self.indexed = indexed
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

  * `id` - the item identifier.
  * `item` - the item.

]]
function FileIterator:next()
  if self:isEOF() then
    return nil
  end

  self.offset = self.offset + 1

  local id

  if self.indexed then
    id = self:_readId()
  else
    id = self.offset
  end

  return id, self:_read()
end

--[[ Read the next line identifier as a string. ]]
function FileIterator:_readId()
  local id = ''

  while true do
    local c = self.file:read(1)

    if c == nil then
      error('empty line ' .. self.offset)
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
