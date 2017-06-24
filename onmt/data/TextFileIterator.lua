require('onmt.data.FileIterator')

--[[ Simple FileIterator that iterates line by line. ]]
local TextFileIterator, parent = torch.class('TextFileIterator', 'FileIterator')

function TextFileIterator:__init(...)
  parent.__init(self, ...)
end

--[[ Read the next line. ]]
function TextFileIterator:_read()
  return self.file:read()
end

return TextFileIterator
