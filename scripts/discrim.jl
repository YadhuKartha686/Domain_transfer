using DrWatson
@quickactivate "Domain_transfer"
using Random
using Metalhead
using Flux
using Flux.Optimise: Optimiser, ExpDecay, update!, ADAM , gpu
using LinearAlgebra
using PyPlot
using MLDatasets
using Statistics
using Images 
using JLD2
using BSON: @save 
using Printf 
using MLUtils
using Random; Random.seed!(1)



model = Chain(
    Chain([
      Conv((7, 7), 1 => 64, pad=3, stride=2, bias=false),  # 9_408 parameters
      BatchNorm(64, relu),            # 128 parameters, plus 128
      MaxPool((3, 3), pad=1, stride=2),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
          BatchNorm(64, relu),        # 128 parameters, plus 128
          Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
          BatchNorm(64),              # 128 parameters, plus 128
        ),
        identity,
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
          BatchNorm(64, relu),        # 128 parameters, plus 128
          Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
          BatchNorm(64),              # 128 parameters, plus 128
        ),
        identity,
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 64 => 128, pad=1, stride=2, bias=false),  # 73_728 parameters
          BatchNorm(128, relu),       # 256 parameters, plus 256
          Conv((3, 3), 128 => 128, pad=1, bias=false),  # 147_456 parameters
          BatchNorm(128),             # 256 parameters, plus 256
        ),
        Chain([
          Conv((1, 1), 64 => 128, stride=2, bias=false),  # 8_192 parameters
          BatchNorm(128),             # 256 parameters, plus 256
        ]),
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 128 => 128, pad=1, bias=false),  # 147_456 parameters
          BatchNorm(128, relu),       # 256 parameters, plus 256
          Conv((3, 3), 128 => 128, pad=1, bias=false),  # 147_456 parameters
          BatchNorm(128),             # 256 parameters, plus 256
        ),
        identity,
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 128 => 256, pad=1, stride=2, bias=false),  # 294_912 parameters
          BatchNorm(256, relu),       # 512 parameters, plus 512
          Conv((3, 3), 256 => 256, pad=1, bias=false),  # 589_824 parameters
          BatchNorm(256),             # 512 parameters, plus 512
        ),
        Chain([
          Conv((1, 1), 128 => 256, stride=2, bias=false),  # 32_768 parameters
          BatchNorm(256),             # 512 parameters, plus 512
        ]),
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 256 => 256, pad=1, bias=false),  # 589_824 parameters
          BatchNorm(256, relu),       # 512 parameters, plus 512
          Conv((3, 3), 256 => 256, pad=1, bias=false),  # 589_824 parameters
          BatchNorm(256),             # 512 parameters, plus 512
        ),
        identity,
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 256 => 512, pad=1, stride=2, bias=false),  # 1_179_648 parameters
          BatchNorm(512, relu),       # 1_024 parameters, plus 1_024
          Conv((3, 3), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
          BatchNorm(512),             # 1_024 parameters, plus 1_024
        ),
        Chain([
          Conv((1, 1), 256 => 512, stride=2, bias=false),  # 131_072 parameters
          BatchNorm(512),             # 1_024 parameters, plus 1_024
        ]),
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
          BatchNorm(512, relu),       # 1_024 parameters, plus 1_024
          Conv((3, 3), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
          BatchNorm(512),             # 1_024 parameters, plus 1_024
        ),
        identity,
      ),
    ]),
    Chain(
      AdaptiveMeanPool((1, 1)),
      MLUtils.flatten,
      Dense(512 => 2, sigmoid),             # 513_000 parameters
    ),
  )

# model = ResNet(18);
# model = Chain(model.layers[1:end-1],AdaptiveMeanPool((1, 1)),
# MLUtils.flatten,Dense(512,2,softmax))
# conv_layer = Conv((7, 7), 1 => 64, pad=3, stride=2, bias=false)

# Define the model
# model = Chain(
#     conv_layer
#     # Add more layers if necessary
# )
model = gpu(model)

# Training hyperparameters 
n_epochs     = 20
# device = gpu
lr = 1f-3
opt = Flux.Optimise.ADAM(lr) |> gpu
loss(x, y) = Flux.binarycrossentropy(x, y) |> gpu

function normalize_images(data)
  num_images, height, width, channels = size(data)
  
  for i in 1:num_images
      min_vals = minimum(data[i, :, :, :], dims=(1, 2, 3))
      max_vals = maximum(data[i, :, :, :], dims=(1, 2, 3))
      data[i, :, :, :] .= (data[i, :, :, :] .- min_vals) ./ (max_vals .- min_vals)
  end
  
  return data
end


#Loading Data
data_path= "/data/CompassShot.jld2"

train_X = jldopen(data_path, "r")["X"]
train_y = jldopen(data_path, "r")["Y"]

test_x = zeros(Float32, 2048, 512, 1:1,301)

for i=1:301
  sigma = 1.0
  test_x[:,:,:,i] = imresize(imfilter(train_X[:,:,i+1499],KernelFactors.gaussian((sigma,sigma))),(2048,512))
end
test_y = train_y[1500:end,:]


train_x = zeros(Float32, 2048, 512, 1:1,1504)

for i=1:1504
  sigma = 1.0
  train_x[:,:,:,i] = imresize(imfilter(train_X[:,:,i],KernelFactors.gaussian((sigma,sigma))),(2048,512))
end

train_y = train_y[1:1504,:]
num_classes = maximum(train_y) + 1

# Perform one-hot encoding using Flux.onehotbatch
onehot_labels = Flux.onehotbatch(train_y, 0:num_classes-1)
onehot_labels_test = Flux.onehotbatch(test_y, 0:num_classes-1)
train_x = normalize_images(train_x)
test_x = normalize_images(test_x)
n_train = size(train_x)[end]
n_test = size(test_x)[end]-1
batch_size=2
n_batches = cld(n_train, batch_size) 
n_batches_test = cld(n_test, batch_size) 
# Training Loop
losslist=[]
accuracylist=[]

losslistpara=[]
accuracylistpara=[]

losslist_test=[]
accuracylist_test=[]
for e=1:n_epochs
  epoch_loss=0.0
  epoch_loss_test=0.0
  correct_predictions=0.0
  correct_predictions_test=0.0
  idx_e = reshape(randperm(n_train), batch_size, n_batches)
    for b = 1:n_batches # batch loop
        x = train_x[:,:,:,idx_e[:,b]]
        x .+= 0.001*randn(Float32, size(x))
        x = x |> gpu
        y = onehot_labels[:,idx_e[:,b],1] |>gpu
        grads = Flux.gradient(Flux.params(model)) do
          y_pred = model(x)
          l = loss(y_pred,y)
        end
        # epoch_loss +=Loss
        Flux.update!(opt,Flux.params(model),grads)
        
        
        print("batch: $b, epoch: $e \n ")
        # Calculate accuracy for this batch
        y_pred = model(x)
        epoch_loss += loss(y_pred,y)
        predictions = argmax(y_pred,dims=1)
        correct_predictions += sum(predictions .== argmax(y, dims=1))
        push!(losslistpara,loss(y_pred,y))
        push!(accuracylistpara,sum(predictions .== argmax(y, dims=1))/2)

        plt.plot(1:b,losslistpara[1:b],label="train")
        plt.xlabel("epoch")
        plt.ylabel("value")
        plt.legend()
        plt.title("Loss per update")
        plt.savefig("/plot/loss batch $b.png")
        plt.close()


        plt.plot(1:b,accuracylistpara[1:b],label="train")
        plt.xlabel("epoch")
        plt.ylabel("value")
        plt.legend()
        plt.title("acc per update")
        plt.savefig("/plot/acc batch $b.png")
        plt.close()

    end

    idx_e_test = reshape(randperm(n_test), batch_size, n_batches_test)  
    for b = 1:n_batches_test # batch loop
      x = test_x[:,:,:,idx_e_test[:,b]]
      x .+= 0.001*randn(Float32, size(x))
      x = x |> gpu
      y = onehot_labels_test[:,idx_e_test[:,b],1] |>gpu
      
      print("batch: $b \n")
      # Calculate accuracy for this batch
      y_pred = model(x)
      epoch_loss_test += loss(y_pred,y)
      predictions = argmax(y_pred,dims=1)
      correct_predictions_test += sum(predictions .== argmax(y, dims=1))
  end
    
    # Calculate average loss for the epoch
    avg_epoch_loss_test = epoch_loss_test / size(idx_e_test, 2)
    accuracy_test = correct_predictions_test / 300

    push!(losslist_test, avg_epoch_loss_test)
    push!(accuracylist_test, accuracy_test)

    avg_epoch_loss = epoch_loss_test / size(idx_e, 2)
    accuracy = correct_predictions / 1504
    push!(losslist, avg_epoch_loss)
    push!(accuracylist, accuracy)
    
    println("Epoch $e, Average Loss: $avg_epoch_loss, Accuracy: $accuracy% , Average Loss test: $avg_epoch_loss_test, Accuracy test: $accuracy_test%\n")

    plt.plot(1:e,losslist[1:e],label="train")
    plt.plot(1:e,losslist_test[1:e],label="test")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.title("Loss per epoch")
    plt.savefig("/plot/loss $e.png")
    plt.close()


    plt.plot(1:e,accuracylist[1:e],label="train")
    plt.plot(1:e,accuracylist_test[1:e],label="test")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.title("acc per epoch")
    plt.savefig("/plot/acc $e.png")
    plt.close()
end

plt.plot(losslist,label="train")
plt.plot(losslist_test,label="test")
plt.xlabel("epoch")
plt.ylabel("value")
plt.legend()
plt.title("Loss per epoch")
plt.savefig("/plot/loss.png")
plt.close()


plt.plot(accuracylist,label="train")
plt.plot(accuracylist_test,label="test")
plt.xlabel("epoch")
plt.ylabel("value")
plt.legend()
plt.title("acc per epoch")
plt.savefig("/plot/acc.png")
plt.close()