--[[ Return the maxLength, sizes, and non-zero count
  of a batch of `seq`s ignoring `ignore` words.
--]]
local function getLength(seq, ignore)
  local sizes = torch.IntTensor(#seq):zero()
  local max = 0
  local sum = 0

  for i = 1, #seq do
    local len = seq[i]:size(1)
    if ignore ~= nil then
      len = len - ignore
    end
    max = math.max(max, len)
    sum = sum + len
    sizes[i] = len
  end
  return max, sizes, sum
end

--[[ Data management and batch creation.

Batch interface [size]:

  * size: number of sentences in the batch [1]
  * sourceLength: max length in source batch [1]
  * sourceSize:  lengths of each source [batch x 1]
  * sourceInput:  left-padded idx's of source (PPPPPPABCDE) [batch x max]
  * sourceInputFeatures: table of source features sequences
  * sourceInputRev: right-padded  idx's of source rev (EDCBAPPPPPP) [batch x max]
  * sourceInputRevFeatures: table of reversed source features sequences
  * targetLength: max length in source batch [1]
  * targetSize: lengths of each source [batch x 1]
  * targetNonZeros: number of non-ignored words in batch [1]
  * targetInput: input idx's of target (SABCDEPPPPPP) [batch x max]
  * targetInputFeatures: table of target input features sequences
  * targetOutput: expected output idx's of target (ABCDESPPPPPP) [batch x max]
  * targetOutputFeatures: table of target output features sequences

 TODO: change name of size => maxlen
--]]
local Batch = torch.class('Batch')

--[[ Create a batch object given aligned sent tables `src` and `tgt`
  (optional). Data format is shown at the top of the file.
--]]
function Batch:__init(src, srcFeatures, srcDomains, tgt, tgtFeatures, tgtDomains)
  if tgt ~= nil then
    assert(#src == #tgt, "source and target must have the same batch size")
  end

  self.size = #src

  self.sourceLength, self.sourceSize = getLength(src)

  local sourceSeq = torch.IntTensor(self.sourceLength, self.size):fill(onmt.Constants.PAD)
  self.sourceInput = sourceSeq:clone()
  self.sourceInputRev = sourceSeq:clone()

  self.sourceInputFeatures = {}
  self.sourceInputRevFeatures = {}

  if #srcFeatures > 0 then
    for _ = 1, #srcFeatures[1] do
      table.insert(self.sourceInputFeatures, sourceSeq:clone())
      table.insert(self.sourceInputRevFeatures, sourceSeq:clone())
    end
  end

  if #srcDomains > 0 then
    self.sourceDomainInput = torch.IntTensor(self.size):fill(onmt.Constants.PAD)
  end

  if tgt ~= nil then
    self.targetLength, self.targetSize, self.targetNonZeros = getLength(tgt, 1)

    local targetSeq = torch.IntTensor(self.targetLength, self.size):fill(onmt.Constants.PAD)
    self.targetInput = targetSeq:clone()
    self.targetOutput = targetSeq:clone()

    self.targetInputFeatures = {}
    self.targetOutputFeatures = {}

    if #tgtFeatures > 0 then
      for _ = 1, #tgtFeatures[1] do
        table.insert(self.targetInputFeatures, targetSeq:clone())
        table.insert(self.targetOutputFeatures, targetSeq:clone())
      end
    end

    if #tgtDomains > 0 then
      self.targetDomainInput = torch.IntTensor(self.size):fill(onmt.Constants.PAD)
    end
  end

  for b = 1, self.size do
    local sourceOffset = self.sourceLength - self.sourceSize[b] + 1
    local sourceInput = src[b]
    local sourceInputRev = src[b]:index(1, torch.linspace(self.sourceSize[b], 1, self.sourceSize[b]):long())

    -- Source input is left padded [PPPPPPABCDE] .
    self.sourceInput[{{sourceOffset, self.sourceLength}, b}]:copy(sourceInput)
    self.sourceInputPadLeft = true

    -- Rev source input is right padded [EDCBAPPPPPP] .
    self.sourceInputRev[{{1, self.sourceSize[b]}, b}]:copy(sourceInputRev)
    self.sourceInputRevPadLeft = false

    for i = 1, #self.sourceInputFeatures do
      local sourceInputFeatures = srcFeatures[b][i]
      local sourceInputRevFeatures = srcFeatures[b][i]:index(1, torch.linspace(self.sourceSize[b], 1, self.sourceSize[b]):long())

      self.sourceInputFeatures[i][{{sourceOffset, self.sourceLength}, b}]:copy(sourceInputFeatures)
      self.sourceInputRevFeatures[i][{{1, self.sourceSize[b]}, b}]:copy(sourceInputRevFeatures)
    end

    if #srcDomains > 0 then
      self.sourceDomainInput[b] = srcDomains[b]
    end

    if tgt ~= nil then
      -- Input: [<s>ABCDE]
      -- Ouput: [ABCDE</s>]
      local targetLength = tgt[b]:size(1) - 1
      local targetInput = tgt[b]:narrow(1, 1, targetLength)
      local targetOutput = tgt[b]:narrow(1, 2, targetLength)

      -- Target is right padded [<S>ABCDEPPPPPP] .
      self.targetInput[{{1, targetLength}, b}]:copy(targetInput)
      self.targetOutput[{{1, targetLength}, b}]:copy(targetOutput)

      for i = 1, #self.targetInputFeatures do
        local targetInputFeatures = tgtFeatures[b][i]:narrow(1, 1, targetLength)
        local targetOutputFeatures = tgtFeatures[b][i]:narrow(1, 2, targetLength)

        self.targetInputFeatures[i][{{1, targetLength}, b}]:copy(targetInputFeatures)
        self.targetOutputFeatures[i][{{1, targetLength}, b}]:copy(targetOutputFeatures)
      end

      if #tgtDomains > 0 then
        self.targetDomainInput[b] = tgtDomains[b]
      end
    end
  end
end

function Batch:getSourceInput(t)
  local input = self.sourceInput[t]

  if #self.sourceInputFeatures > 0 then
    input = { input }
    for j = 1, #self.sourceInputFeatures do
      table.insert(input, self.sourceInputFeatures[j][t])
    end
  end

  if self.sourceDomainInput then
    if type(input) ~= 'table' then
      input = { input }
    end
    table.insert(input, self.sourceDomainInput)
  end

  return input
end

function Batch:getTargetInput(t)
  local input = self.targetInput[t]

  if #self.targetInputFeatures > 0 then
    input = { input }
    for j = 1, #self.targetInputFeatures do
      table.insert(input, self.targetInputFeatures[j][t])
    end
  end

  if self.targetDomainInput then
    if type(input) ~= 'table' then
      input = { input }
    end
    table.insert(input, self.targetDomainInput)
  end

  return input
end

function Batch:getTargetOutput(t)
  -- If a regular input, return word id, otherwise a table with features.
  local outputs = { self.targetOutput[t] }
  for j = 1, #self.targetOutputFeatures do
    table.insert(outputs, self.targetOutputFeatures[j][t])
  end
  return outputs
end

return Batch
