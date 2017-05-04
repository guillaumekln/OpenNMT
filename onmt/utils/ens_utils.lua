local function convert_table_2_hash(tab)
  local tds = require 'tds'
  local var = tds.Hash()
  for k, v in pairs(tab) do
    if(type(v) == "table")
    then
      var[k] = convert_table_2_hash(v)
    else
      if(type(v) == "userdata") -- process torch.CudaTensor
      then
        local temp = tds.Hash()
        temp[1] = "userdata"
        temp[2] = torch.serialize(v)
        var[k] = temp
      else
        var[k] = v
      end
    end
  end
  
  return var
end

local function convert_hash_2_table(has)
  local var = {}
  for k, v in pairs(has) do
    if(type(v) == "cdata")
    then
      if( v[1] ~= nil and v[1] == "userdata")
      then
        -- restore torch.CudaTensor
        var[k] = torch.deserialize(v[2])
      else
        var[k] = convert_hash_2_table(v)
      end
    else
      var[k] = v
    end
  end
  
  return var
end


function srcFeat()
  return g_datatype == 'feattext'
end

function buildInput(tokens)
  local data = {}
  if g_datatype == 'feattext' then
    data.vectors = torch.Tensor(tokens)
  else
    local words, features = onmt.utils.Features.extract(tokens)

    data.words = words

    if #features > 0 then
      data.features = features
    end
  end

  return data
end

function buildInputGold(tokens)
  local data = {}

  local words, features = onmt.utils.Features.extract(tokens)

  data.words = words

  if #features > 0 then
    data.features = features
  end

  return data
end

function buildOutput(data)
  return table.concat(onmt.utils.Features.annotate(data.words, data.features), ' ')
end

return {
  convert_table_2_hash = convert_table_2_hash,
  convert_hash_2_table = convert_hash_2_table,
  srcFeat = srcFeat,
  buildInput = buildInput,
  buildInputGold = buildInputGold,
  buildOutput = buildOutput
}
