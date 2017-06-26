require('onmt.data.DataTransformer')

local tds = require('tds')
local Constants = require('onmt.Constants')

--[[ Converts tokens to ids. ]]
local StreamsToIds, parent = torch.class('StreamsToIds', 'DataTransformer')

--[[ Creates a new StreamsToIds.

Parameters:

  * `vocabs` - a table of `Dict`, one for each stream.

]]
function StreamsToIds:__init(vocabs)
  parent.__init(self)
  self.vocabs = vocabs
end

--[[ Transform streams of labels to streams of ids .

Parameters:

  * `streams` - a table of sequence of labels.

Returns:

  * `ids` - a table of `IntTensor`.

]]
function StreamsToIds:transform(streams)
  local ids = tds.Vec()

  for i = 1, #streams do
    ids:insert(self.vocabs[i]:toIds(streams[i], Constants.UNK_WORD))
  end

  return ids
end

--[[ Transform `ids` to streams. ]]
function StreamsToIds:reverse(ids)
  local streams = tds.Vec()

  for i = 1, #ids do
    streams:insert(self.vocabs[i]:toLabels(ids[i]))
  end

  return streams
end

return StreamsToIds
