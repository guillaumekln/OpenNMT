function TextPreprocessor:__init(dicts, withStartAndStop, featuresGenerator)
  self.dicts = dicts
  self.featuresGenerator = featuresGenerator

  self.metaTokens = { onmt.Constants.UNK_WORD }
  if withStartAndStop then
    onmt.utils.Table.append(self.metaTokens, { onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD })
  end
end

function TextPreprocessor:next(file)
  local line = file:read()
  local tokens = onmt.utils.String.split(line)

  local words, features = onmt.utils.Features.extract(tokens)

  local data = {}
  data.length = #tokens
  data.input = self.dicts.words:convertToIdx(words, table.unpack(self.metaTokens))
  data.pruned = data.input:eq(onmt.Constants.UNK):sum() / data.input:size(1)

  if #self.dicts.features > 0 then
    data.meta = self.featuresGenerator(self.dicts.features, features)
  end

  return data
end
