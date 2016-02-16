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
mlp:add(nn.Linear(1666, 512))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(512, 256))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(256, 128))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(128, 1))
mlp:cuda()

loss = nn.MSECriterion() 
loss:cuda()
--mlp:add(nn.LogSoftMax())
--loss = nn.ClassNLLCriterion()

-- Configuring optimizer
--local optim_state = {
--   learningRate = 0.01,
--   momentum = 0.6,--0.1,
--   weightDecay = 0.0005--1e-5
--}
--sdg
--optimState = {
--   learningRate = 0.01,
--   momentum = 0.6, --0.1
--   weightDecay = 0.0005 --1e-5
--}

--lbfgs
optim_state_lbfgs = {
--	learningRate = 0.05,
	maxIter = 90,
--	nCorrection = 10
	config.verbose=true
}

w,dE_dw = mlp:getParameters()
--dE_dw:cuda()
print("Starting gradient descent from 'optim' on GPU...")
function  cuda_eval(w)
    dE_dw:zero() -- Обновляем градиенты
    local Y = mlp:forward(input)
    local E = loss:forward(Y,output)
    local dE_dy = loss:backward(Y,output)
    mlp:backward(input,dE_dy)
    return E, dE_dw
	--return E, dE_dy --для пробы, возможно неправильно
end



timer = torch.Timer();

	_,fw = optim.lbfgs(cuda_eval, w, optim_state_lbfgs);

cutorch.synchronize()
print(string.format('Success! Average iteration time was %f', timer:time().real))
print('Success!')
torch.cat(mlp:forward(input),output)
#fw


