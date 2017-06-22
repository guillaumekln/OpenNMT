local String = require('onmt.utils.String')

--[[ Splits text and word features. ]]
local TextSplitter, parent = torch.class('TextSplitter', 'DataTransformer')

function TextSplitter:__init(tokenSeparator, streamSeparator)
  parent.__init(self)
  self.tokenSeparator = tokenSeparator or ' '
  self.streamSeparator = streamSeparator or '￨'
end

function TextSplitter:transform(text)
  local tokens = String.split(String.strip(text), self.tokenSeparator)

  local streams = {}

  for t = 1, #tokens do
    local fields = String.split(tokens[t], self.streamSeparator)

    if #streams == 0 then
      for _ = 1, #fields do
        table.insert(streams, {})
      end
    else
      assert(#fields == #streams, 'all words must have the same number of features')
    end

    for i = 1, #fields do
      table.insert(streams[i], fields[i])
    end
  end

  return streams
end

function TextSplitter:reverse(streams)
  local tokens = {}

  for i = 1, #streams do
    for j = 1, #streams[i] do
      if tokens[j] then
        tokens[j] = tokens[j] .. self.streamSeparator .. streams[i][j]
      else
        tokens[j] = streams[i][j]
      end
    end
  end

  return table.concat(tokens, self.tokenSeparator)
end

return TextSplitter
