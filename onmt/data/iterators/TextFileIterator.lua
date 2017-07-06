require('onmt.data.iterators.FileIterator')

--[[ Simple `FileIterator` that iterates line by line. ]]
local TextFileIterator, parent = torch.class('TextFileIterator', 'FileIterator')

function TextFileIterator:__init(...)
  parent.__init(self, ...)
end

--[[ Reads the next line. ]]
function TextFileIterator:_readItem()
  return self.file:read()
end

return TextFileIterator
