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
using ImageTransformations
using SlimPlotting
using InvertibleNetworks
using MLUtils
using BSON 
# using Wavelets
using StatsBase
using Distributions
using Images

#### DATA LOADING #####
using MLDatasets

plot_path = "."



# Load in training data
range_digA = [1]
range_digB = [7]
# load in training data
train_x, train_y = MNIST.traindata()

# grab the digist to train on 
inds = findall(x -> x in range_digA, train_y)
train_yA = train_y[inds]
train_xA = zeros(Float32,16,16,1,5923)
for i=1:5923
  train_xA[:,:,:,i] = reverse(imresize(train_x[:,:,inds[i]],(16,16)),dims=1)
  train_xA[:,:,1,i] = rotr90(train_xA[:,:,1,i],1)
  # train_xA[:,:,:,i] = train_x[:,:,inds[i]]
end

inds = findall(x -> x in range_digB, train_y)
train_yB = train_y[inds]
train_xB = zeros(Float32,16,16,1,5851)
for i=1:5851

  train_xB[:,:,:,i] = reverse(imresize(train_x[:,:,inds[i]],(16,16)),dims=1)
  train_xB[:,:,1,i] = rotr90(train_xB[:,:,1,i],1)
  # train_xB[:,:,:,i] = train_x[:,:,inds[i]] 
end

nx,ny = 16, 16
N = nx*ny;


# Define the generator and discriminator networks


device = gpu #GPU does not accelerate at this small size. quicker on cpu
lr     = 1f-5
epochs = 30
batch_size = 1
low = 0.5f0

# Architecture parametrs
chan_x = 1; chan_y = 1; L = 2; K = 10; n_hidden = 128 # Number of hidden channels in convolutional residual blocks

# Create network
G = NetworkConditionalGlow(chan_x, chan_y, n_hidden,  L, K; split_scales=true,activation=SigmoidLayer(low=low,high=1.0f0)) |> device;

model = Chain(
    Conv((3, 3), 1=>64, relu;pad=1),
    x -> maxpool(x, (2,2)),
    Conv((3, 3), 64=>32, relu;pad=1),
    x -> maxpool(x, (2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(512, 1),
    sigmoid
)


# Define the loss functions
function Dissloss(real_output, fake_output)
  real_loss = mean(Flux.binarycrossentropy.(real_output, 1f0))
  fake_loss = mean(Flux.binarycrossentropy.(fake_output, 0f0))
  return 0.5f0*(real_loss + fake_loss)
end

function Genloss(fake_output,x,y) 
  return mean(Flux.binarycrossentropy.(fake_output, 1f0)) + 0*Flux.mse(y|> device,x)
end


# Initialize networks and optimizers
generator = G |> gpu
discriminatorA = gpu(model)
discriminatorB = gpu(model)

# generator = G
# discriminatorA = model
# discriminatorB = model

opt_adam = "adam"
clipnorm_val = 5f0
# optimizer_g = Flux.Optimiser(ClipNorm(clipnorm_val), ADAM(lr))
lrd = 1f-5
# optimizer_da = Flux.ADAM(lrd)
# optimizer_db = Flux.ADAM(lrd)
genloss=[]
dissloss = []
imgs = 32
n_train = 1024
n_test = 4500
n_batches = cld(n_train,imgs)
YA = ones(Float32,16,16,1,imgs) + randn(Float32,16,16,1,imgs) ./1000
YB = ones(Float32,16,16,1,imgs) .*7 + randn(Float32,16,16,1,imgs) ./1000


optimizer_g = Flux.Optimiser(ClipNorm(clipnorm_val),ExpDecay(lr, .99f0, n_batches*lr_step, 1f-6), ADAM(lr))
optimizer_da = Flux.Optimiser(ClipNorm(clipnorm_val),ExpDecay(lr, .99f0, n_batches*lr_step, 1f-6), ADAM(lrd))
optimizer_db = Flux.Optimiser(ClipNorm(clipnorm_val),ExpDecay(lr, .99f0, n_batches*lr_step, 1f-6), ADAM(lrd))

lossnrm      = []; logdet_train = []; 
factor = 1f0

n_epochs     = 5000
for e=1:n_epochs# epoch loop
  epoch_loss_diss=0.0
  epoch_loss_gen=0.0
  idx_eA = reshape(randperm(n_train), imgs, n_batches)
  idx_eB = reshape(randperm(n_train), imgs, n_batches)
  for b = 1:n_batches # batch loop
        @time begin
          ############# Loading domain A data ############## 
    
          XA = train_xA[:, :, :, idx_eA[:,b]] + randn(Float32,16,16,1,imgs) ./10000
          XB = train_xB[:, :, :, idx_eA[:,b]] + randn(Float32,16,16,1,imgs) ./10000
          X = cat(XA, XB,dims=4)
          Y = cat(YA, YB,dims=4)
     
          Zx, Zy, lgdet = generator.forward(X|> device, Y|> device)  #### concat so that network normalizes ####
          
 

          ######## interchanging conditions to get domain transferred images during inverse call #########

          ZyA = Zy[:,:,:,1:imgs]
          ZyB = Zy[:,:,:,imgs+1:end]

          ZxA = Zx[:,:,:,1:imgs]
          ZxB = Zx[:,:,:,imgs+1:end]

          Zy1 = cat(ZyB,ZyA,dims=4)

     

          fake_images,invcall = generator.inverse(Zx|>device,Zy1)  ###### generating images #######
  
          ####### getting fake images from respective domain ########

          fake_imagesAfromB = fake_images[:,:,:,imgs+1:end]
          fake_imagesBfromA = fake_images[:,:,:,1:imgs]

          ####### discrim training ########
          # for ii=1:2
            dA_grads = Flux.gradient(Flux.params(discriminatorA)) do
                real_outputA = discriminatorA(XA|> device)
                fake_outputA = discriminatorA(fake_imagesAfromB|> device)
                lossA = Dissloss(real_outputA, fake_outputA)
            end
            Flux.Optimise.update!(optimizer_da, Flux.params(discriminatorA),dA_grads)  #### domain A discrim ####

            
            dB_grads = Flux.gradient(Flux.params(discriminatorB)) do
                real_outputB = discriminatorB(XB|> device)
                fake_outputB = discriminatorB(fake_imagesBfromA|> device)
                lossB = Dissloss(real_outputB, fake_outputB)
            end
            Flux.Optimise.update!(optimizer_db, Flux.params(discriminatorB),dB_grads)  #### domain B discrim ####
          # end
          ## minlog (1-D(fakeimg)) <--> max log(D(fake)) + norm(Z)
                    
          gsA = gradient(x -> Genloss(discriminatorA(x|> device),x,XA), fake_imagesAfromB)[1]  #### getting gradients wrt A fake ####
          gsB = gradient(x -> Genloss(discriminatorB(x|> device),x,XB), fake_imagesBfromA)[1]  #### getting gradients wrt B fake ####
          

          gs = cat(gsB,gsA,dims=4)
          generator.backward_inv(((gs ./ factor)|>device), fake_images, invcall;) #### updating grads wrt image ####
          generator.backward(Zx / imgs*2, Zx, Zy;)
        
          for p in get_params(generator)
              Flux.update!(optimizer_g,p.data,p.grad)
          end
          clear_grad!(generator) #### updating generator ####

          #loss calculation for printing
          fake_outputA = discriminatorA(fake_imagesAfromB|> device)
          fake_outputB = discriminatorB(fake_imagesBfromA|> device)
          real_outputA = discriminatorA(XA|> device)
          real_outputB = discriminatorB(XB|> device)

          lossAd = Dissloss(real_outputA, fake_outputA)  #### log(D(real)) + log(1 - D(fake)) ####
          lossBd = Dissloss(real_outputB, fake_outputB)  #### log(D(real)) + log(1 - D(fake)) ####
          lossA = Genloss(fake_outputA,fake_imagesAfromB,fake_imagesAfromB)  #### log(1 - D(fake)) + mse ####
          lossB = Genloss(fake_outputB,fake_imagesAfromB,fake_imagesAfromB)  #### log(1 - D(fake)) + mse ####
          f_all = norm(Zx)^2

          loss = lossA + lossB #+ ml

          append!(lossnrm, f_all / (imgs*2*N))  # normalize by image size and batch size
          append!(logdet_train, (-lgdet) / N) # logdet is internally normalized by batch size
          append!(genloss, loss)  # normalize by image size and batch size
          append!(dissloss, (lossAd+lossBd) ) # logdet is internally normalized by batch size


          epoch_loss_diss += (lossAd+lossBd)
          epoch_loss_gen += loss

          println("Iter: epoch=", e, "/", n_epochs,":batch = ",b,
          "; Genloss=", loss, 
            "; genloss = ",  loss+(f_all / (imgs*2*N)), 
              "; dissloss = ", (lossAd+lossBd) , 
              "; f l2 = ",  lossnrm[end], 
              "; lgdet = ", logdet_train[end], "\n")

          Base.flush(Base.stdout)
        end
        
        if mod(e,1)==0 && mod(b,n_batches)==0
          avg_epoch_lossd = epoch_loss_diss / size(idx_eA, 2)
          avg_epoch_lossg= epoch_loss_gen / size(idx_eA, 2)
          push!(genloss, avg_epoch_lossg)
          push!(dissloss, avg_epoch_lossd)
        end

        if mod(e,5)==0 && mod(b,n_batches)==0
          x = train_xA[:,:,:,4500:4999]
          x .+= 0.001*randn(Float32, size(x))
          x = x |> gpu
          y = transpose(ones(Float32,500))|>gpu
          # Calculate accuracy for this batch
          y_pred = discriminatorA(x)
          correct_predictions_testa = sum(abs.(y_pred-y))/500

          println("l2norm of Da: ",correct_predictions_testa)

          x = train_xB[:,:,:,4500:4999]
          x .+= 0.001*randn(Float32, size(x))
          x = x |> gpu
          y = transpose(ones(Float32,500))|>gpu
          # Calculate accuracy for this batch
          y_pred = discriminatorB(x)
          correct_predictions_testb = sum(abs.(y_pred-y))/500

          println("l2norm of Db: ",correct_predictions_testb)
       end

        if mod(e,10) == 0 && mod(b,n_batches)==0
          plt.plot(lossnrm)
          plt.title("loss $e")
          plt.savefig("../plots/Shot_rec_df/lossnorm$e.png")
          plt.close()
          plt.plot(logdet_train)
          plt.title("logdet $e")
          plt.savefig("../plots/Shot_rec_df/logdet$e.png")
          plt.close()
      
          plt.plot(1:e,dissloss[1:e],label = "Diss")
          plt.plot(1:e,genloss[1:e], label = "Gen")
          plt.title("ganloss $e")
          plt.legend()
          plt.savefig("../plots/Shot_rec_df/ganloss$e.png")
          plt.close()
        end

    end
    XA = zeros(Float32 , 16,16,1,imgs)
    XB = zeros(Float32 , 16,16,1,imgs)
    XA[:,:,:,1:imgs] = train_xA[:,:,:,n_test:n_test-1+imgs] +  randn(Float32,16,16,1,imgs) ./1000
    XB[:,:,:,1:imgs] = train_xB[:,:,:,n_test:n_test-1+imgs] + randn(Float32,16,16,1,imgs) ./1000

    X = cat(XA, XB,dims=4)
    Y = cat(YA, YB,dims=4)
    Zx, Zy, lgdet = generator.forward(X|> device, Y|> device)  #### concat so that network normalizes ####


              ######## interchanging conditions to get domain transferred images during inverse call #########

    ZyA = Zy[:,:,:,1:imgs]
    ZyB = Zy[:,:,:,imgs+1:end]

    Zy = cat(ZyB,ZyA,dims=4)

    fake_images,invcall = generator.inverse(Zx|>device,Zy)  ###### generating images #######

              ####### getting fake images from respective domain ########

    fake_imagesAfromBt = fake_images[:,:,:,imgs+1:end]
    fake_imagesBfromAt = fake_images[:,:,:,1:imgs]

    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(3,2,1)
    ax1.imshow(XB[:,:,:,1],vmin = 0,vmax = 1)
    ax1.title.set_text("data test ")


    ax2 = fig.add_subplot(3,2,2)
    ax2.imshow(fake_imagesAfromBt[:,:,1,1]|>cpu,vmin = 0,vmax = 1)
    ax2.title.set_text("digit pred 1 from 7 ")


    ax3 = fig.add_subplot(3,2,3)
    ax3.imshow(XB[:,:,:,2],vmin = 0,vmax = 1)
    ax3.title.set_text("data test ")


    ax4 = fig.add_subplot(3,2,4)
    ax4.imshow(fake_imagesAfromBt[:,:,1,2]|>cpu,vmin = 0,vmax = 1)
    ax4.title.set_text("digit pred 1 from 7 ")



    ax5 = fig.add_subplot(3,2,5)
    ax5.imshow(XB[:,:,:,3],vmin = 0,vmax = 1)
    ax5.title.set_text("data test ")


    ax6 = fig.add_subplot(3,2,6)
    ax6.imshow(fake_imagesAfromBt[:,:,1,3]|>cpu,vmin = 0,vmax = 1)
    ax6.title.set_text("digit pred 1 from 7 ")


    fig.savefig("../plots/Shot_rec_df/number one test $e.png")
    plt.close(fig)


    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(3,2,1)
    ax1.imshow(XA[:,:,:,1],vmin = 0,vmax = 1)
    ax1.title.set_text("data test ")


    ax2 = fig.add_subplot(3,2,2)
    ax2.imshow(fake_imagesBfromAt[:,:,1,1]|>cpu,vmin = 0,vmax = 1)
    ax2.title.set_text("digit pred 7 from 1 ")

    ax3 = fig.add_subplot(3,2,3)
    ax3.imshow(XA[:,:,:,2],vmin = 0,vmax = 1)
    ax3.title.set_text("data test ")


    ax4 = fig.add_subplot(3,2,4)
    ax4.imshow(fake_imagesBfromAt[:,:,1,2]|>cpu,vmin = 0,vmax = 1)
    ax4.title.set_text("digit pred 7 from 1 ")


    ax5 = fig.add_subplot(3,2,5)
    ax5.imshow(XA[:,:,:,4],vmin = 0,vmax = 1)
    ax5.title.set_text("data test ")


    ax6 = fig.add_subplot(3,2,6)
    ax6.imshow(fake_imagesBfromAt[:,:,1,4]|>cpu,vmin = 0,vmax = 1)
    ax6.title.set_text("digit pred 7 from 1 ")


    fig.savefig("../plots/Shot_rec_df/number seven test $e.png")
    plt.close(fig)

    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(3,2,1)
    ax1.imshow(Zx[:,:,1,1]|>cpu,vmin = 0,vmax = 1)
    ax1.title.set_text("latent space of digit 1 ")


    ax2 = fig.add_subplot(3,2,2)
    ax2.imshow(Zx[:,:,1,1+imgs]|>cpu,vmin = 0,vmax = 1)
    ax2.title.set_text("latent space of digit 1  ")

    fig.savefig("../plots/Shot_rec_df/latent test $e.png")
    plt.close(fig)


    if mod(e,10) == 0
      Params = get_params(generator) |> cpu;
      save_dict = @strdict e L K n_hidden lr
      @tagsave(
           "../plots/Shot_rec_df/"*savename(save_dict, "jld2"; digits=6),
           save_dict;
           safe=true
      )
    end

end


print("done training!!!")
