-- рабочая сеть для CPU
require 'torch'
require 'nn'
require 'csvigo'
require 'nnx'
require 'optim'



trainPath = '/home/dmitry/nn/Spectr_MFNN.csv'

trainData= csvigo.load({path= trainPath, mode='raw', separator=';' })
-- testData = torch.load(testPath,'ascii')
csv_tensor = torch.Tensor(trainData)

input= csv_tensor:sub(1,30,2,1667)
output = csv_tensor:sub(1,30,1,1)

-- define the mlp
mlp = nn.Sequential()

mlp:add(nn.Reshape(1666))
mlp:add(nn.Linear(1666, 256))
mlp:add(nn.Tanh())
--mlp:add(nn.Linear(300, 300))
--mlp:add(nn.Tanh())
mlp:add(nn.Linear(256, 128))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(128, 1))


loss = nn.MSECriterion() 

--mlp:add(nn.LogSoftMax())
--loss = nn.ClassNLLCriterion()
--maxIter =100;
-- Configuring optimizer
optim_state = {
   learningRate = 0.01,
   momentum = 0.6,--0.1,
   weightDecay = 0.0005,--1e-5
	maxIter = 100,
	verbose =true
}

w,dE_dw = mlp:getParameters()

function eval_E()--w)
    dE_dw:zero() -- Обновляем градиенты
    local Y = mlp:forward(input)
    local E = loss:forward(Y,output)
    local dE_dy = loss:backward(Y,output)
    mlp:backward(input,dE_dy)
    return E, dE_dw
	--return E, dE_dy --для пробы, возможно неправильно
end
-- optim_method = optim.sgd
mlp:reset()
optim.lbfgs(eval_E, w, optim_state);
mlp:forward(input)



timer = torch.Timer();

for i=1,maxIter do
	_,fw = optim.lbfgs(eval_E, w, optim_state);
	if i%(torch.floor(maxIter/10))==0 then print(string.format('MSE = %f',fw[1])) end
end
cutorch.synchronize()
print(string.format('Success! Average iteration time was %f',timer:time().real/maxIter))


--- конец
--- GPU---------------------------------------------------------
require 'torch'
require 'nn'
require 'csvigo'
require 'nnx'
require 'optim'
require 'cutorch'
require 'cunn'



trainPath = '/home/dmitry/nn/Spectr_MFNN.csv'

trainData= csvigo.load({path= trainPath, mode='raw', separator=';' })
-- testData = torch.load(testPath,'ascii')
csv_tensor = torch.Tensor(trainData)

input= csv_tensor:sub(1,30,2,1667)
input = input:cuda()
output = csv_tensor:sub(1,30,1,1)
output=output:cuda()

-- define the mlp
mlp = nn.Sequential()

mlp:add(nn.Reshape(1666))
mlp:add(nn.Linear(1666, 256))
mlp:add(nn.Tanh())
--mlp:add(nn.Linear(300, 300))
--mlp:add(nn.Tanh())
mlp:add(nn.Linear(256, 128))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(128, 1))
mlp:cuda()

loss = nn.MSECriterion() 
loss = loss:cuda()
--mlp:add(nn.LogSoftMax())
--loss = nn.ClassNLLCriterion()

-- Configuring optimizer
--local optim_state = {
--   learningRate = 0.01,
--   momentum = 0.6,--0.1,
--   weightDecay = 0.0005--1e-5
--}
--lbfgs
optimState = {
      learningRate = 0.01,
      maxIter = 400,
      nCorrection = 10
   }
print("Starting gradient descent from 'optim' on GPU...")
function cuda_eval_E()--w)
    dE_dw:zero() -- Обновляем градиенты
    local Y = mlp:forward(input)
    local E = loss:forward(Y,output)
    local dE_dy = loss:backward(Y,output)
    mlp:backward(input,dE_dy)
    return E, dE_dw
	--return E, dE_dy --для пробы, возможно неправильно
end
mlp:reset()
optim.lbfgs(cuda_eval_E, w, optim_state);
cutorch.synchronize()
mlp:forward(input)

timer = torch.Timer();
maxIter =50;
for i=1,maxIter do
	_,fw = optim.lbfgs(cuda_eval_E, w, optim_state);
	if i%(torch.floor(maxIter/10))==0 then print(string.format('MSE = %f',fw[1])) end
end
cutorch.synchronize()
print(string.format('Success! Average iteration time was %f',timer:time().real/maxIter))

--batchSize = 128
--local yt = torch.Tensor(batchSize)
--yt = yt:cuda()


-- Создаём специальные переменные: веса нейросети и их градиенты
 


-- Затем в цикле обучения вызываем
optim.sgd(eval_E, w, optim_state)
optim.lbfgs(eval_E, w, optim_state)



print("Starting gradient descent from 'optim' on GPU...")
timer = torch.Timer();
maxIter =50;
for i=1,maxIter do
	_,fw = optim.lbfgs(eval_E, w, optim_state);
	if i%(torch.floor(maxIter/10))==0 then print(string.format('MSE = %f',fw[1])) end
end
cutorch.synchronize()
print(string.format('Success! Average iteration time was %f',timer:time().real/maxIter))

