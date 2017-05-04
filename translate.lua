require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('translate.lua')

local options = {
  {
    '-src', '',
    [[Source sequences to translate.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-tgt', '',
    [[Optional true target sequences.]]
  },
  {
    '-output', 'pred.txt',
    [[Output file.]]
  },
  {
    '-idx_files', false,
    [[If set, source and target files are 'key value' with key match between source and target.]]
  }
}

cmd:setCmdLineOptions(options, 'Data')

onmt.translate.Translator.declareOpts(cmd)
onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

cmd:text('')
cmd:text('**Other options**')
cmd:text('')

cmd:option('-time', false, [[Measure average translation time.]])

local function reportScore(name, scoreTotal, wordsTotal)
  _G.logger:info(name .. " AVG SCORE: %.2f, " .. name .. " PPL: %.2f",
                 scoreTotal / wordsTotal,
                 math.exp(-scoreTotal/wordsTotal))
end

local opt = cmd:parse(arg)
_G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)
_G.profiler = onmt.utils.Profiler.new()
local globalLogger = _G.logger
local globalProfiler = _G.profiler
local datatype

local function check_available_models()
  local suffix = 1
  
  while true
  do
    local model_name = opt.model .. '.' .. suffix
    print('checking ' .. model_name .. '...')
    
		local is_file_exist = io.open(model_name)
		if(is_file_exist ~= nil) then
      io.close(is_file_exist)
    else
      print(model_name .. ' does not exist ...')
      
      break
		end
    
    suffix = suffix + 1
	end
  
  return suffix-1
end

local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')

local status, tds = pcall(require, 'tds')
tds = status and tds or nil
if not status then return end

local nmodels = check_available_models()

if(nmodels == 0)
then
  print("model does not exist, exit")
  return
end

local nthread = nmodels + 2   -- one for main method, one for ensemble
local atomic = tds.AtomicCounter()
local atomic_ens = tds.AtomicCounter()

-- synchronize between main thread and two models
local m = {}
local c = {}
local m_id = {}
local c_id = {}
for i = 1, nmodels do
  m[i] = threads.Mutex()
  c[i] = threads.Condition()
  m_id[i] = m[i]:id()
  c_id[i] = c[i]:id()
end
local m_main = threads.Mutex()
local c_main = threads.Condition()
local m_main_id = m_main:id()
local c_main_id = c_main:id()

-- synchronize between ensemble calculation thread and two models
local m_ens = {}
local c_ens = {}
local m_ens_id = {}
local c_ens_id = {}
for i = 1, nmodels do
  m_ens[i] = threads.Mutex()
  c_ens[i] = threads.Condition()
  m_ens_id[i] = m_ens[i]:id()
  c_ens_id[i] = c_ens[i]:id()
end
local m_main_ens = threads.Mutex()
local c_main_ens = threads.Condition()
local m_main_ens_id = m_main_ens:id()
local c_main_ens_id = c_main_ens:id()

-- synchronize between two models using same gpu
local m_gpu = {}
local m_gpu_id = {}
for i = 1, nmodels do
  m_gpu[i] = threads.Mutex()
  m_gpu_id[i] = m_gpu[i]:id()
end

-- transfer data between main thread and two models
local g_input = tds.Hash()
local g_result = tds.Hash()

-- transfer data between ensemble calculation thread and two models
local g_input_vec = tds.Hash()
local g_result_vec = tds.Hash()

local pool = threads.Threads(
   nthread,
    function()
    require('cunn')
    require('nngraph')
    require('onmt.init')
    end,
    function(threadid)
    print('starting a new thread/state number ' .. threadid)
    g_opt = opt   -- share "opt" in all the threads and can be modified
    g_nmodels = nmodels
    _G.logger = globalLogger
    _G.profiler = globalProfiler
    g_datatype = datatype
   end
)

local function main(c, m_main, c_main)
  local ens_utils = require 'onmt.utils.ens_utils'
  
  local srcReader = onmt.utils.FileReader.new(opt.src, opt.idx_files, ens_utils.srcFeat())
  local srcBatch = {}
  local srcIdBatch = {}

  local goldReader
  local goldBatch

  local withGoldScore = opt.tgt:len() > 0

  if withGoldScore then
    goldReader = onmt.utils.FileReader.new(opt.tgt, opt.idx_files)
    goldBatch = {}
  end

  local outFile = io.open(opt.output, 'w')

  local sentId = 1
  local batchId = 1

  local predScoreTotal = 0
  local predWordsTotal = 0
  local goldScoreTotal = 0
  local goldWordsTotal = 0

  local timer
  if opt.time then
    timer = torch.Timer()
    timer:stop()
    timer:reset()
  end

  local ens_utils = require 'onmt.utils.ens_utils'
  while true do
    local srcSeq, srcSeqId = srcReader:next()

    local goldOutputSeq
    if withGoldScore then
      goldOutputSeq = goldReader:next()
    end

    if srcSeq ~= nil then
      table.insert(srcBatch, ens_utils.buildInput(srcSeq))
      table.insert(srcIdBatch, srcSeqId)

      if withGoldScore then
        table.insert(goldBatch, ens_utils.buildInputGold(goldOutputSeq))
      end
    elseif #srcBatch == 0 then
      break
    end

    if srcSeq == nil or #srcBatch == opt.batch_size then
      if opt.time then
        timer:resume()
      end

      for i = 1, nmodels do
        g_input[i] = ens_utils.convert_table_2_hash(srcBatch)

        c[i]:signal()
      end
      
      while true do 
        if(atomic:get() == nmodels) then
          atomic:set(0)
          break
        else
          c_main:wait(m_main)
        end
      end
      
--      local results = translator:translate(srcBatch, goldBatch)
      local results = ens_utils.convert_hash_2_table(g_result[1])

      if opt.time then
        timer:stop()
      end

      for b = 1, #results do
        if (srcBatch[b].words and #srcBatch[b].words == 0) then
          _G.logger:warning('Line ' .. sentId .. ' is empty.')
          outFile:write('\n')
        else
          if srcBatch[b].words then
            _G.logger:info('SENT %d: %s', sentId, ens_utils.buildOutput(srcBatch[b]))
          else
            _G.logger:info('FEATS %d: IDX - %s - SIZE %d', sentId, srcIdBatch[b], srcBatch[b].vectors:size(1))
          end

          if withGoldScore then
            _G.logger:info('GOLD %d: %s', sentId, ens_utils.buildOutput(goldBatch[b]), results[b].goldScore)
            _G.logger:info("GOLD SCORE: %.2f", results[b].goldScore)
            goldScoreTotal = goldScoreTotal + results[b].goldScore
            goldWordsTotal = goldWordsTotal + #goldBatch[b].words
          end
          if opt.dump_input_encoding then
            outFile:write(sentId, ' ', table.concat(torch.totable(results[b]), " "), '\n')
          else
            for n = 1, #results[b].preds do
              local sentence = ens_utils.buildOutput(results[b].preds[n])
              outFile:write(sentence .. '\n')
              if n == 1 then
                predScoreTotal = predScoreTotal + results[b].preds[n].score
                predWordsTotal = predWordsTotal + #results[b].preds[n].words

                if #results[b].preds > 1 then
                  _G.logger:info('')
                  _G.logger:info('BEST HYP:')
                end
              end

              if #results[b].preds > 1 then
                _G.logger:info("[%.2f] %s", results[b].preds[n].score, sentence)
              else
                _G.logger:info("PRED %d: %s", sentId, sentence)
                _G.logger:info("PRED SCORE: %.2f", results[b].preds[n].score)
              end
            end
          end
        end
        _G.logger:info('')
        sentId = sentId + 1
      end

      if srcSeq == nil then
        break
      end

      batchId = batchId + 1
      srcBatch = {}
      srcIdBatch = {}
      if withGoldScore then
        goldBatch = {}
      end
      collectgarbage()
    end
  end

  if opt.time then
    local time = timer:time()
    local sentenceCount = sentId-1
    _G.logger:info("Average sentence translation time (in seconds):\n")
    _G.logger:info("avg real\t" .. time.real / sentenceCount .. "\n")
    _G.logger:info("avg user\t" .. time.user / sentenceCount .. "\n")
    _G.logger:info("avg sys\t" .. time.sys / sentenceCount .. "\n")
  end

  if opt.dump_input_encoding == false then
    reportScore('PRED', predScoreTotal, predWordsTotal)

    if withGoldScore then
      reportScore('GOLD', goldScoreTotal, goldWordsTotal)
    end
  end
  outFile:close()
  _G.logger:shutDown()
end

local function cal_ens()
  local ens_utils = require 'onmt.utils.ens_utils'

  local out = {}
  for i = 1, nmodels do
    table.insert(out, ens_utils.convert_hash_2_table(g_input_vec[i]))
  end

  -- do not need to modify, if ensemble 2+ models 
  local out_ens = {}
  
  for j = 1, #out[1] do
    out_ens[j] = torch.DoubleTensor():cuda()
    
    out_ens[j]:resize(out[1][j]:size()):copy(out[1][j])
  end
  
  for i = 2, #out do
    for j = 1, #out_ens do
      out_ens[j] = torch.add(out_ens[j], out[i][j])
    end        
  end
  
  for j = 1, #out_ens do
    out_ens[j] = torch.div(out_ens[j], #out)
  end        

  return out_ens
end

pool:specific(true)

for i=1, nmodels do
  pool:addjob(i,
    function()
      require('onmt.init')
      onmt.utils.Cuda.init(opt, i)

      return __threadid
    end
  )
end  

for i=1, nmodels do
  pool:addjob(i,
    function()
      require('onmt.init')
      local translator = onmt.translate.Translator.new(opt, i)
      
      local t_threads = require 'threads'
      local m = {}
      local c = {}
      for j = 1, nmodels do
        m[j] = t_threads.Mutex(m_id[j])
        c[j] = t_threads.Condition(c_id[j])
      end
      local m_main = t_threads.Mutex(m_main_id)
      local c_main = t_threads.Condition(c_main_id)

      local m_ens = {}
      local c_ens = {}
      for j = 1, nmodels do
        m_ens[j] = t_threads.Mutex(m_ens_id[j])
        c_ens[j] = t_threads.Condition(c_ens_id[j])
      end
      local m_main_ens = t_threads.Mutex(m_main_ens_id)
      local c_main_ens = t_threads.Condition(c_main_ens_id)

      local m_gpu = {}
      for j = 1, nmodels do
        m_gpu[j] = t_threads.Mutex(m_gpu_id[j])
      end
      
      require 'cutorch'
      require 'cunn'
      
      m[i]:lock()
      m_ens[i]:lock()

      if((i == 1 and nmodels >= 5) or (i == 2 and nmodels >= 6) or
          (i == 3 and nmodels >= 7) or (i == 4 and nmodels >= 8))
      then
        m_gpu[i]:lock()
      end

      local ens_utils = require 'onmt.utils.ens_utils'

      while true do
        print("== thread " .. __threadid)
        
        atomic:inc()
        --print(atomic:get())
        if(atomic:get() == nmodels)
        then
          c_main:signal()
        end
        
        c[i]:wait(m[i])

        if(atomic:get() == 99)
        then
          break
        end

        if((i == 1 and nmodels >= 5) or (i == 2 and nmodels >= 6) or
            (i == 3 and nmodels >= 7) or (i == 4 and nmodels >= 8))
        then
          m_gpu[i]:lock()
        end
        
        srcBatch = ens_utils.convert_hash_2_table(g_input[i])
        
        local result = translator:translate(srcBatch, goldBatch, c_ens, m_ens, c_main_ens, g_input_vec, atomic_ens, g_result_vec, m_gpu)
        --print(result)
        
        g_result[i] = ens_utils.convert_table_2_hash(result)

        if((i == 5 and nmodels >= 5) or (i == 6 and nmodels >= 6) or
            (i == 7 and nmodels >= 7) or (i == 8 and nmodels >= 8))
        then
          m_gpu[i-4]:unlock()
        end

        collectgarbage()
        collectgarbage()
      end  

      print("finish thread " .. __threadid)
    end
  )
end

pool:addjob(nmodels+1,
  function()
    require('onmt.init')
    local t_threads = require 'threads'
    local m = {}
    local c = {}
    for j = 1, nmodels do
      m[j] = t_threads.Mutex(m_id[j])
      c[j] = t_threads.Condition(c_id[j])
    end
    local m_main = t_threads.Mutex(m_main_id)
    local c_main = t_threads.Condition(c_main_id)
    
    local c_main_ens = t_threads.Condition(c_main_ens_id)

    require 'cutorch'
    require 'cunn'
      
    m_main:lock()
    while true do 
      if(atomic:get() == nmodels) then
        atomic:set(0)
        break
      else
        c_main:wait(m_main)
      end
    end
    
    main(c, m_main, c_main)
    
    -- set close signal
    atomic:set(99)
    for j = 1, nmodels do
      c[j]:signal()
    end

    -- set close signal
    atomic_ens:set(99)
    c_main_ens:signal()
    
    print("finish thread " .. __threadid)
  end
)

pool:addjob(nmodels+2,
  function()
    require('onmt.init')
    local t_threads = require 'threads'
    local m_ens = {}
    local c_ens = {}
    for j = 1, nmodels do
      m_ens[j] = t_threads.Mutex(m_ens_id[j])
      c_ens[j] = t_threads.Condition(c_ens_id[j])
    end
    local m_main_ens = t_threads.Mutex(m_main_ens_id)
    local c_main_ens = t_threads.Condition(c_main_ens_id)

    require 'cutorch'
    require 'cunn'

    local count = cutorch.getDeviceCount()
    cutorch.setDevice(count)
    
    local ens_utils = require 'onmt.utils.ens_utils'
    
    m_main_ens:lock()
    while true do
      while true do 
        if(atomic_ens:get() == nmodels) then
          atomic_ens:set(0)
          break
        else
          c_main_ens:wait(m_main_ens)

          if(atomic_ens:get() == 99)
          then
            break
          end
        end
      end

      if(atomic_ens:get() == 99)
      then
        break
      end
      
      local out_ens = cal_ens()
      
      for j = 1, nmodels do
        g_result_vec[j] = ens_utils.convert_table_2_hash(out_ens)
      
        c_ens[j]:signal()
      end
    end
    
    print("finish thread " .. __threadid)
   
  end
)

pool:synchronize()

pool:terminate()

for i = 1, nmodels do
  m[i]:free()
  c[i]:free()
  m_ens[i]:free()
  c_ens[i]:free()
  
  m_gpu[i]:free()
end
m_main:free()
c_main:free()

m_main_ens:free()
c_main_ens:free()
