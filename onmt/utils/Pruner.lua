local Pruner = torch.class('Pruner')

local options = {
  {'-prune_at', '', [[Comma-separated list of epoch after which to prune paramters.]]},
  {'-prune_ratio', '',  [[Comma-separated list of pruning ratio to apply.]]}
}

function Pruner.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Pruning')
end

function Pruner:__init()
end

function Pruner:reset(params, ratio)
  local _, indices = torch.topk(params, math.floor(ratio * params:size(1)))
  params:indexFill(1, indices, 0)
  self.negmask = torch.eq(params, 0)
end

function Pruner:maskGradients(grads)
  if self.negmask then
    grads:maskedFill(self.negmask, 0)
  end
end

--[[ Convert `m` parameters to a CSR sparse representation. --]]
function Pruner.toSparse(m)
  if m.weight then
    local negmask = torch.eq(m.weight, 0)
    local values = m.weight.new(negmask:nElement() - negmask:sum())
    local offset = 0

    m.weightRowOffsets = torch.IntTensor(m.weight:size(1) + 1)
    m.weightCols = torch.IntTensor(values:size(1))

    for i = 1, m.weight:size(1) do
      m.weightRowOffsets[i] = offset
      for j = 1, m.weight:size(2) do
        if negmask[i][j] == 0 then
          offset = offset + 1
          m.weightCols[offset] = j
          values[offset] = m.weight[i][j]
        end
      end
    end

    m.weightRowOffsets[m.weightRowOffsets:size(1)] = offset
    m.weightSize = torch.IntTensor({m.weight:size(1), m.weight:size(2)})
    m.weight = values
  end

  if m.bias then
    local negmask = torch.eq(m.bias, 0)
    local values = m.bias.new(negmask:nElement() - negmask:sum())
    local offset = 0

    m.biasRows = torch.IntTensor(values:size(1))

    for i = 1, m.bias:size(1) do
      if negmask[i] == 0 then
        offset = offset + 1
        m.biasRows[offset] = i
        values[offset] = m.bias[i]
      end
    end

    m.biasSize = torch.IntTensor({m.bias:size(1)})
    m.bias = values
  end

  return m
end

return Pruner
