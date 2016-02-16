-- Include modules/libraries
require 'torch'
require 'nn'


print('')
print('============================================================')
print('Constructing dataset')
print('')



-- The data are stored in a csv file 'example-logistic-regression.csv'
-- and read with the csvigo package (torch-pkg install csvigo)

require 'csvigo'

-- Reading CSV files can be tricky. This code uses the csvigo package for this:
loaded = csvigo.load('example-logistic-regression.csv')

regression = nn.Sequential()
regression:add(nn.Linear(1666))
regression:add(nn.Linear(1) )
loss = nn.ClassNLLCriterion()




require 'torch'
require 'nn'
require 'optim'
require 'xlua'

require 'cutorch'
require 'cunn'
require 'csvigo'

trainPath = '/home/dmitry/nn/Spectr_MFNN.csv'
-- testPath = '/home/dmitry/nn/test_32x32.t7'
-- trainData = torch.load(trainPath,'ascii')
trainData= csvigo.load({path= trainPath, mode='raw', separator=';' })
-- testData = torch.load(testPath,'ascii')
csv_tensor = torch.Tensor(trainData)

input= csv_tensor:sub(1,30,2,1667)
target = csv_tensor:sub(1,30,1,1)


-- normalize data
--std = trainData.data[{ {1, trainSize} }]:std()
--trainData.data[{ {1, trainSize} }]:div(std)
-- testData.data[{ {1, testSize} }]:div(std)


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
--mlp:add(nn.Tanh())

mlp:add(nn.LogSoftMax())

loss = nn.ClassNLLCriterion()

optimState = {
   learningRate = 1e-2,
   momentum = 0.9,--0.1,
   weightDecay = 0.0005--1e-5
}

batchSize = 128


--criterion = nn.MSECriterion()  
--trainer = nn.StochasticGradient(mlp, criterion)
--trainer.learningRate = 0.01
--trainer:train(dataset)

for i = 1,29 do


  -- feed it to the neural network and the criterion
  criterion:forward(mlp:forward(input1[i]), output1[i])

  -- train over this example in 3 steps
  -- (1) zero the accumulation of the gradients
  mlp:zeroGradParameters()
  -- (2) accumulate gradients
  mlp:backward(input, criterion:backward(mlp.output, output))
  -- (3) update parameters with a 0.01 learning rate
  mlp:updateParameters(0.01)
end



