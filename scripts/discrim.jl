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
using SlimPlotting
using BSON: @save 
using Printf 
using MLUtils
using Random; Random.seed!(1)

# model = Chain(
#   Conv((3, 3), 1 => 64, relu, pad=(1, 1), stride=(1, 1)), 
#   BatchNorm(64),
#   Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
#   BatchNorm(64),
#   MaxPool((2,2)),

#   Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
#   BatchNorm(128),
#   Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
#   BatchNorm(128),
#   MaxPool((2,2)),

#   Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
#   BatchNorm(256),
#   Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
#   BatchNorm(256),
#   Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
#   BatchNorm(256),
#   MaxPool((2,2)),

#   Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
#   BatchNorm(512),
#   Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
#   BatchNorm(512),
#   Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
#   BatchNorm(512),
#   MaxPool((2,2)),

#   Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
#   BatchNorm(512),
#   Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
#   BatchNorm(512),
#   Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
#   BatchNorm(512),
#   MaxPool((2,2)),

#   x -> reshape(x, :, size(x, 4)),
#   Dense(8192, 4096, relu),  
#   Dropout(0.5),

#   Dense(4096, 2048, relu),
#   Dropout(0.5),
  
#   Dense(2048, 1),
#   sigmoid
#   ) 

function discriminator_block(in_channels, out_channels, stride, padding, use_norm=true)
  layers = [Conv((4, 4), in_channels => out_channels, stride=stride, pad=padding, leakyrelu(0.2))]
  if use_norm
      push!(layers, BatchNorm(out_channels, relu))
  end
  return layers
end

function PatchGANDiscriminator()
  return Chain(
      discriminator_block(1, 64, 2, 1, false),  # First layer, no normalization, input has 1 channel
      discriminator_block(64, 128, 2, 1),
      discriminator_block(128, 256, 2, 1),
      Conv((4, 4), 256 => 1, stride=1, pad=1)  # Final layer reduces to 1 channel per patch
      # No batch norm and activation in the final layer
  )
end

model = PatchGANDiscriminator()

# model = Chain(
#   # First convolutional layer
#   Conv((7, 7), 1=>32, relu, pad=(3,3), stride=1),
#   x -> maxpool(x, (2,2)), # Aggressive pooling to reduce dimensions
  
#   # Second convolutional layer
#   Conv((5, 5), 32=>64, relu, pad=(2,2), stride=1),
#   x -> maxpool(x, (2,2)), # Further reduction
  
#   # Third convolutional layer
#   Conv((5, 5), 64=>128, relu, pad=1),
#   x -> maxpool(x, (2,2)), # Final pooling to reduce to a very low dimension
#   # Fourth convolutional layer
#   Conv((3, 3), 128=>64, relu, pad=1),
#   x -> maxpool(x, (2,2)), # Final pooling to reduce to a very low dimension

#   Conv((3, 3), 64=>64, relu, pad=1),
#   x -> maxpool(x, (2,2)), # Final pooling to reduce to a very low dimension
  
#   # Flatten the output of the last convolutional layer before passing it to the dense layer
#   Flux.flatten,
  
#   # Fully connected layer
#   Dense(576, 512, relu),
  
#   # Output layer for binary classification
#   Dense(512, 1),
#   sigmoid
# )

model = gpu(model)

# Training hyperparameters 
n_epochs     = 20
# device = gpu
lr = 1f-3
opt = Flux.Optimise.ADAM(lr) |> gpu
loss(x, y) = mean(Flux.binarycrossentropy.(x, y))

function patch_bce_loss(y_pred, y_true)
  # Reshape if necessary to ensure we can compare predictions to labels
  y_pred_flat = reshape(y_pred, :)
  y_true_flat = reshape(y_true, :)
  
  # Compute BCE loss for each element
  loss = binarycrossentropy.(y_pred_flat, y_true_flat)
  
  # Return the mean loss
  return mean(loss)
end



nx,ny = 128, 128
N = nx*ny;

data_path= "../data/CompassShotmid_els.jld2"
# datadir(CompassShot.jld2)
train_X = jldopen(data_path, "r")["X"]
train_y = jldopen(data_path, "r")["Y"]
  
train_xA = zeros(Float32, nx, ny, 1,2160)
train_xB = zeros(Float32, nx, ny, 1,2160)

  
for i=1:2160
    sigma = 1.0
    train_xA[:,:,:,i] = imresize(imfilter(train_X[:,:,i],KernelFactors.gaussian((sigma,sigma))),(nx,ny))
    train_xB[:,:,:,i] = imresize(imfilter(train_X[:,:,2160+i],KernelFactors.gaussian((sigma,sigma))),(nx,ny))
end

# mina=minimum(train_xA)
# minb=minimum(train_xB)
train_xA = train_xA.* (abs.(train_xA) .> 5e-6)
train_xB = train_xB.* (abs.(train_xB) .> 5e-6)

maxa=maximum(abs.(train_xA))
maxb=maximum(abs.(train_xB))

train_xA = (train_xA )./(maxa) 
train_xB = (train_xB )./(maxb) 

plot_sdata(train_xB[:,:,1,1]|>cpu,(14.06,4.976),perc=95,vmax=0.02,cbar=true)
    plt.title("data test vel+den ")
    plt.savefig("../plots/Shot_rec_df/vel+den data test1.png")
    plt.close()
# Perform one-hot encoding using Flux.onehotbatch
# train_x = normalize_images(train_x)
# test_x = normalize_images(test_x)
idx = reshape(randperm(4000), 4000, 1)

train_x= zeros(Float32,(nx,ny,1,4000))
train_y= zeros(Float32,4000)
train_x[:,:,:,1:2000] = train_xA[:,:,:,1:2000]
train_x[:,:,:,2001:4000] = train_xB[:,:,:,1:2000]
train_y[2001:4000] .= 1.0

train_y = train_y[idx]
train_x = train_x[:,:,:,idx]

test_x = train_x[:,:,:,3008:4000]
train_x = train_x[:,:,:,1:3008]
test_y = train_y[3008:4000]
train_y = train_y[1:3008]

n_train = size(train_x)[end]
n_test = size(test_x)[end]-1
batch_size=16
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
  correct_predictions_testa=0.0
  idx_e = reshape(randperm(n_train), batch_size, n_batches)
    for b = 1:n_batches # batch loop
        x = train_x[:,:,:,idx_e[:,b]]
        # x .+= 0.001*randn(Float32, size(x))
        x = x |> gpu
        y = transpose(train_y[idx_e[:,b]])|>gpu
        grads = Flux.gradient(Flux.params(model)) do
          y_pred = model(x)
          l = patch_bce_loss(y_pred,y)
        end
        # epoch_loss +=Loss
        Flux.update!(opt,Flux.params(model),grads)
        
        
        print("batch: $b, epoch: $e \n ")
        # Calculate accuracy for this batch
        y_pred = model(x)
        epoch_loss += patch_bce_loss(y_pred,y)
        correct_predictions += sum(round.(y_pred) .== y)
    end

    idx_e_test = reshape(randperm(n_test), batch_size, n_batches_test)  
    for b = 1:n_batches_test # batch loop
      x = test_x[:,:,:,idx_e_test[:,b]]
      # x .+= 0.001*randn(Float32, size(x))
      x = x |> gpu
      y = transpose(test_y[idx_e_test[:,b]])|>gpu
      
      print("batch: $b \n")
      # Calculate accuracy for this batch
      y_pred = model(x)
      epoch_loss_test += patch_bce_loss(y_pred,y)
      correct_predictions_test += sum(round.(y_pred) .== y)
      correct_predictions_testa += sum(abs.(y_pred-y))
      
  end
    
    # Calculate average loss for the epoch
    avg_epoch_loss_test = epoch_loss_test / size(idx_e_test, 2)
    accuracy_test = correct_predictions_test / (n_batches_test*batch_size)
    println("l2norm of Da: ",correct_predictions_testa/ (n_batches_test*batch_size))
    push!(losslist_test, avg_epoch_loss_test)
    push!(accuracylist_test, accuracy_test)

    avg_epoch_loss = epoch_loss_test / size(idx_e, 2)
    accuracy = correct_predictions / (n_batches*batch_size)
    push!(losslist, avg_epoch_loss)
    push!(accuracylist, accuracy)
    
    println("Epoch $e, Average Loss: $avg_epoch_loss, Accuracy: $accuracy% , Average Loss test: $avg_epoch_loss_test, Accuracy test: $accuracy_test%\n")

    plt.plot(1:e,losslist[1:e],label="train")
    plt.plot(1:e,losslist_test[1:e],label="test")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.title("Loss per epoch")
    plt.savefig("../plots/loss $e.png")
    plt.close()


    plt.plot(1:e,accuracylist[1:e],label="train")
    plt.plot(1:e,accuracylist_test[1:e],label="test")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.title("acc per epoch")
    plt.savefig("../plots/acc $e.png")
    plt.close()
end

plt.plot(losslist,label="train")
plt.plot(losslist_test,label="test")
plt.xlabel("epoch")
plt.ylabel("value")
plt.legend()
plt.title("Loss per epoch")
plt.savefig("../plots/loss.png")
plt.close()


plt.plot(accuracylist,label="train")
plt.plot(accuracylist_test,label="test")
plt.xlabel("epoch")
plt.ylabel("value")
plt.legend()
plt.title("acc per epoch")
plt.savefig("../plots/acc.png")
plt.close()