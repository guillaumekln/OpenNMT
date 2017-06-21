local tds

local TextPreprocessor, parent = torch.class('TextPreprocessor', 'Preprocessor')

function TextPreprocessor:__init(dicts, maxLength)
  self.dicts = dicts
  parent.__init(self, maxLength)
end

function TextPreprocessor:consume(file)
  local line = file:read()
  return line
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

  local featuresIds
  local success

  if not tds then
    success, tds = pcall(require, 'tds')
    if not success then
      featuresIds = {}
      tds = nil
    else
      featuresIds = tds.Vec()
    end
  else
    featuresIds = tds.Vec()
  end

  return featuresIds
end

return TextPreprocessor
