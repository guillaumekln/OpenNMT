require('onmt.data.transformers.DataTransformer')

local tds = require('tds')
local String = require('onmt.utils.String')

--[[ Splits text and word features. ]]
local TextSplitter, parent = torch.class('TextSplitter', 'DataTransformer')

--[[ Creates a new TextSplitter.

Parameters:

  * `tokenSeparator` - the token or timestep separator.
  * `streamSeparator` - the separator of each input stream (a.k.a word features).

]]
function TextSplitter:__init(tokenSeparator, streamSeparator)
  parent.__init(self)
  self.tokenSeparator = tokenSeparator or ' '
  self.streamSeparator = streamSeparator or '￨'
end

--[[ Transform raw text to streams of tokens.

Parameters:

  * `text` - a string.

Returns:

  * `streams` - a table of sequences of labels.

]]
function TextSplitter:transform(text)
  text = String.strip(text)

  local tokens = String.split(text, self.tokenSeparator)
  local streams = tds.Vec()

  for t = 1, #tokens do
    local fields = String.split(tokens[t], self.streamSeparator)

    if #streams == 0 then
      for _ = 1, #fields do
        streams:insert(tds.Vec())
      end
    else
      assert(#fields == #streams, 'all words must have the same number of features')
    end

    for i = 1, #fields do
      streams[i]:insert(fields[i])
    end
  end

  return streams
end

--[[ Transform `streams` to plain text. ]]
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
