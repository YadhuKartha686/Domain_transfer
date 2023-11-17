using DrWatson
@quickactivate "Domain_transfer"

using Flux
using LinearAlgebra
using Random
using Metalhead
using JLD2
using Statistics
using ImageQualityIndexes 
using PyPlot
using SlimPlotting
using InvertibleNetworks
using MLUtils
using BSON 
# using Wavelets
using StatsBase
using Distributions
using Images

#### DATA LOADING #####

data_path= "../data/CompassShot.jld2"
# datadir(CompassShot.jld2)
train_X = jldopen(data_path, "r")["X"]
train_y = jldopen(data_path, "r")["Y"]
  
test_xA = zeros(Float32, 2048, 512, 1,148)
test_xB = zeros(Float32, 2048, 512, 1,148)
train_xA = zeros(Float32, 2048, 512, 1,752)
train_xB = zeros(Float32, 2048, 512, 1,752)

train_YA = ones(Float32,2048,512,1,752)
train_YB = ones(Float32,2048,512,1,752).*2
test_YA = ones(Float32,2048,512,1,148)
test_YB = ones(Float32,2048,512,1,148).*2



indices_of_A = findall(x -> x == 0.0, train_y[:,1])
indices_of_B = findall(x -> x == 1.0, train_y[:,1])

for i=1:148
    sigma = 1.0
    test_xA[:,:,:,i] = imresize(imfilter(train_X[:,:,indices_of_A[752+i]],KernelFactors.gaussian((sigma,sigma))),(2048,512))
    test_xB[:,:,:,i] = imresize(imfilter(train_X[:,:,indices_of_B[752+i]],KernelFactors.gaussian((sigma,sigma))),(2048,512))
end 

  
for i=1:752
    sigma = 1.0
    train_xA[:,:,:,i] = imresize(imfilter(train_X[:,:,indices_of_A[i]],KernelFactors.gaussian((sigma,sigma))),(2048,512))
    train_xB[:,:,:,i] = imresize(imfilter(train_X[:,:,indices_of_B[i]],KernelFactors.gaussian((sigma,sigma))),(2048,512))
end


batch_size = 8
nx,ny = 2048, 512
N = nx*ny;

n_train = 752

n_batches = cld(n_train,batch_size)


# Define the generator and discriminator networks

n_epochs     = 100
device = gpu
lr = 1f-4
lr_step   = 10
lr_rate = 0.75f0
clipnorm_val = 10f0
noise_lev_x  = 0.005f0
noise_lev_y  = randn(Float32,(2048,512,1,batch_size))./1000
split_scales = true

#User params
save_every   = 5
plot_every   = 1

# Number of samples to test conditional mean quality metric on.
n_condmean = 32
posterior_samples = 32 


K = 9
L = 5# l=5 is better
n_hidden = 64
low = 0.5f0
n_in = 1
n_out = 1
G = NetworkConditionalGlow(n_out, n_in, n_hidden,  L, K; split_scales=split_scales, activation=SigmoidLayer(low=low,high=1.0f0));


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
      Dense(512 => 1, sigmoid),             # 513_000 parameters
    ),
  )



# Define the loss functions
function Dissloss(real_output, fake_output)
  real_loss = mean(Flux.binarycrossentropy.(real_output, 1f0))
  fake_loss = mean(Flux.binarycrossentropy.(fake_output, 0f0))
  return (real_loss + fake_loss)
end

Genloss(fake_output) = mean(Flux.binarycrossentropy.(fake_output, 1f0))


# Initialize networks and optimizers
generator = G |> gpu
discriminatorA = gpu(model)
discriminatorB = gpu(model)

# generator = G
# discriminatorA = model
# discriminatorB = model

generator_params = Flux.params(generator)
discriminatorA_params = Flux.params(discriminatorA)
discriminatorB_params = Flux.params(discriminatorA)

opt_adam = "adam"
optimizer_g = Flux.Optimiser(ExpDecay(lr, lr_rate, n_batches*lr_step, 1f-6), ClipNorm(clipnorm_val), ADAM(lr))
optimizer_da = Flux.Optimiser(ExpDecay(lr, lr_rate, n_batches*lr_step, 1f-6), ClipNorm(clipnorm_val), ADAM(lr))
optimizer_db = Flux.Optimiser(ExpDecay(lr, lr_rate, n_batches*lr_step, 1f-6), ClipNorm(clipnorm_val), ADAM(lr))
genloss=[]
dissloss = []
# Main training loop
for e=1:n_epochs# epoch loop
    epoch_loss_diss=0.0
    epoch_loss_gen=0.0
    idx_eA = reshape(randperm(n_train), batch_size, n_batches)
    idx_eB = reshape(randperm(n_train), batch_size, n_batches)
    for b = 1:n_batches # batch loop
    	@time begin
        
            XA = train_xA[:, :, :, idx_eA[:,b]];
            YA = train_YA[:, :, :, idx_eA[:,b]];
            XA .+= noise_lev_x*randn(Float32, size(XA));
            YA = YA + noise_lev_y;

            XB = train_xB[:, :, :, idx_eB[:,b]];
            YB = train_YB[:, :, :, idx_eB[:,b]];
            XB .+= noise_lev_x*randn(Float32, size(XB));
            YB = YB + noise_lev_y;  
      


            ZxB, ZyB, lgdetb = generator.forward(XB|> device, YB|> device)
            ZxA, ZyA, lgdeta = generator.forward(XA|> device, YA|> device)


            fake_imagesAfromB,invcallA = generator.inverse(ZxB,ZyA)
            fake_imagesBfromA,invcallB = generator.inverse(ZxA,ZyB)
          

            dA_grads = Flux.gradient(discriminatorA_params) do
                real_outputA = discriminatorA(XA|> device)
                fake_outputA = discriminatorA(fake_imagesAfromB|> device)
                lossA = Dissloss(real_outputA, fake_outputA)
            end
            Flux.Optimise.update!(optimizer_da, discriminatorA_params,dA_grads)

            
            dB_grads = Flux.gradient(discriminatorB_params) do
                real_outputB = discriminatorB(XB|> device)
                fake_outputB = discriminatorB(fake_imagesBfromA|> device)
                lossB = Dissloss(real_outputB, fake_outputB)
            end
            Flux.Optimise.update!(optimizer_db, discriminatorB_params,dB_grads)

            ## minlog (1-D(fakeimg)) <--> max log(D(fake)) + norm(Z)
                      
            grad_fake_imagesAfromB = gradient(x -> Genloss(discriminatorA(x|> device)), fake_imagesAfromB)[1]
            grad_fake_imagesBfromA = gradient(x -> Genloss(discriminatorB(x|> device)), fake_imagesBfromA)[1]

            generator.backward_inv(grad_fake_imagesAfromB, fake_imagesAfromB, invcallA;)
            generator.backward_inv(grad_fake_imagesBfromA, fake_imagesBfromA, invcallB;)

            for p in get_params(generator)
              if !isnothing(p.grad)
                Flux.update!(optimizer_g,p.data,p.grad)
              end
            end
            clear_grad!(generator)

            #loss calculation for printing
            fake_outputA = discriminatorA(fake_imagesAfromB|> device)
            fake_outputB = discriminatorB(fake_imagesBfromA|> device)
            real_outputA = discriminatorA(XA|> device)
            real_outputB = discriminatorB(XB|> device)

            lossAd = Dissloss(real_outputA, fake_outputA)
            lossBd = Dissloss(real_outputB, fake_outputB)
            lossA = Genloss(fake_outputA)
            lossB = Genloss(fake_outputB)
            ml = (norm(ZxA)^2 + norm(ZxB)^2)/(N*batch_size)
            loss = lossA + lossB + ml

            epoch_loss_diss += (lossAd+lossBd)/2
            epoch_loss_gen += loss

            println("Iter: epoch=", e, "/", n_epochs, ", batch=", b, "/", n_batches, 
	            "; genloss = ",  loss, 
                "; dissloss = ", (lossAd+lossBd)/2 , "\n")

            Base.flush(Base.stdout)

        end    
    end
    avg_epoch_lossd = epoch_loss_diss / size(idx_eA, 2)
    avg_epoch_lossg= epoch_loss_gen / size(idx_eA, 2)
    push!(genloss, avg_epoch_lossg)
    push!(dissloss, avg_epoch_lossd)
    plt.plot(1:e,genloss[1:e])
    plt.title("genloss $b")
    plt.savefig("../plots/Shot_rec/genloss$e.png")
    plt.close()
    plt.plot(1:e,dissloss[1:e])
    plt.title("dissloss $b")
    plt.savefig("../plots/Shot_rec/dissloss$e.png")
    plt.close()
    
    if n_epochs % 1 == 0
        
        # Optionally, generate and save a sample image during training
        ZxB, ZyB, lgdetb = generator.forward(train_xA[:,:,:,1:8]|> device, train_YA[:,:,:,1:8]|> device)
        ZxA, ZyA, lgdeta = generator.forward(train_xB[:,:,:,1:8]|> device, train_YB[:,:,:,1:8]|> device)


        fake_imagesAfromB,_ = generator.inverse(ZxB,ZyA)
        fake_imagesBfromA,_ = generator.inverse(ZxA,ZyB)

        plot_sdata(XB[:,:,:,1],(0.8,1),vmax=0.04f0,perc=95,cbar=true)
        plt.title("Shot record (vel+den) $e")
        plt.savefig("../plots/Shot_rec/vel+dentrain$e.png")
        plt.close()

        plot_sdata(XA[:,:,:,1],(0.8,1),vmax=0.04f0,perc=95,cbar=true)
        plt.title("Shot record pred vel) $e")
        plt.savefig("../plots/Shot_rec/veltrain$e.png")
        plt.close()
        # Save or visualize the generated_image as needed

        plot_sdata(fake_imagesAfromB[:,:,:,1],(0.8,1),vmax=0.04f0,perc=95,cbar=true)
        plt.title("Shot record pred ( vel from vel+den) $e")
        plt.savefig("../plots/Shot_rec/vel+den$e.png")
        plt.close()

        plot_sdata(fake_imagesBfromA[:,:,:,1],(0.8,1),vmax=0.04f0,perc=95,cbar=true)
        plt.title("Shot record pred ( vel+den from vel) $e")
        plt.savefig("../plots/Shot_rec/vel$e.png")
        plt.close()
        # Save or visualize the generated_image as needed
    end
end