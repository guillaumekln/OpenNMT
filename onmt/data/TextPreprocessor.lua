local tds

local TextPreprocessor, parent = torch.class('TextPreprocessor', 'Preprocessor')

function TextPreprocessor:__init(dicts, maxLength)
  self.dicts = dicts
  parent.__init(self, maxLength)
end

function TextPreprocessor:consume(file)
  return file:read()
end

function TextPreprocessor:tokenize(line)
  local tokens = onmt.utils.String.split(line, ' ')
  local words, features = onmt.utils.Features.extract(tokens)
  return words, features
end

function TextPreprocessor:process(line)
  local words, features = self:tokenize(line)

  local data = {}
  data.length = #words
  data.input = self:convertWords(words)
  data.pruned = data.input:eq(onmt.Constants.UNK):sum() / data.length

  if #self.dicts.features > 0 then
    data.meta = self:convertFeatures(features)
  end

  return data
end

function TextPreprocessor:convertWords(words)
  return self.dicts.words:convertToIdx(words, onmt.Constants.UNK_WORD)
end

function TextPreprocessor:convertFeatures(features)
  local featuresIds = self:_initFeatures(features)

  for j = 1, #self.dicts.features do
    featuresIds[j] = self.dicts.features[j]:convertToIdx(features[j], onmt.Constants.UNK_WORD)
  end

  return featuresIds
end

function TextPreprocessor:_initFeatures(features)
  assert(#self.dicts.features == #features,
         "expected " .. #self.dicts.features .. " features, got " .. #features)

  local success

  if tds == nil then
    success, tds = pcall(require, 'tds')
    if not success then
      tds = false
    end
  end

  return tds and tds.Vec() or {}
end

return TextPreprocessor
