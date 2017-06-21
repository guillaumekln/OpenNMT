--[[ Separate words and features (if any). ]]
local function extract(tokens)
  local words = {}
  local features = {}
  local numFeatures = nil

  for t = 1, #tokens do
    local field = onmt.utils.String.split(tokens[t], '￨')
    local word = field[1]

    if word:len() > 0 then
      table.insert(words, word)

      if numFeatures == nil then
        numFeatures = #field - 1
      else
        assert(#field - 1 == numFeatures,
               'all words must have the same number of features')
      end

      if #field > 1 then
        for i = 2, #field do
          if features[i - 1] == nil then
            features[i - 1] = {}
          end
          table.insert(features[i - 1], field[i])
        end
      end
    end
  end
  return words, features, numFeatures or 0
end

--[[ Reverse operation: attach features to tokens. ]]
local function annotate(tokens, features)
  if not features or #features == 0 then
    return tokens
  end

  for i = 1, #tokens do
    for j = 1, #features do
      tokens[i] = tokens[i] .. '￨' .. features[j][i]
    end
  end

  return tokens
end

return {
  extract = extract,
  annotate = annotate
}
