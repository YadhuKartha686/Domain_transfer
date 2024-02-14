using DrWatson
@quickactivate "Juliatest_1.10"
using InvertibleNetworks
using Random
using Flux
using Flux.Optimise: Optimiser, ExpDecay, update!, ADAM
using LinearAlgebra
using PyPlot
using MLDatasets
using Images
# Function to visualize latent space in 2d instead
# Because network outputs a 1D vector
# Random seed

range_digA = [1]

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


device = gpu #GPU does not accelerate at this small size. quicker on cpu
lr     = 1f-5
epochs = 30
batch_size = 1
low = 0.5f0

# Architecture parametrs
chan_x = 1; chan_y = 1; L = 3; K = 5; n_hidden = 64 # Number of hidden channels in convolutional residual blocks

# Create network
G = NetworkConditionalGlow(chan_x, chan_y, n_hidden,  L, K; split_scales=true,activation=SigmoidLayer(low=low,high=1.0f0)) |> device;

#opt = Optimiser(ExpDecay(lr, .99f0, nbatches*lr_step, 1f-6), ADAM(lr))
opt = Optimiser(ADAM(lr))

# Training log keeper
lossnrm      = []; logdet_train = []; 
generator = G|>gpu
X = train_xA[:,:,:,1:4]
Y = randn(Float32 , (16,16,1,4))
Znoise = randn(Float32 , (16,16,1,4))


plot_every = 2

for e=1:100

        Zx, Zy, lgdet = generator.forward(X|> device, Y|> device)  #### concat so that network normalizes ####

        append!(lossnrm, norm(Zx)^2 / (4*256))  # normalize by image size and batch size
        append!(logdet_train, (-lgdet) / 256) # logdet is internally normalized by batch size
        fake_images,invcall = generator.inverse(Znoise |>device, Zy)
        generator.backward(Zx / 2*2, Zx, Zy;)
        

        print("Iter: epoch=", e, "/", nepochs, 
            "; f l2 = ",   norm(Zx)^2 / (4*256), 
            "; lgdet = ", (-lgdet) / 256, "\n")


        for p in get_params(generator)
            Flux.update!(opt, p.data, p.grad)
        end
        clear_grad!(generator)

        Base.flush(Base.stdout)

    if mod(e,plot_every) ==0
        ############# Test Image generation
        # Evaluate network on test dataset
          ###### generating images #######
        
        fig = plt.figure(figsize=(15, 15))
        ax1 = fig.add_subplot(3,2,1)
        ax1.imshow(X[:,:,:,1],vmin = 0,vmax = 1)
        ax1.title.set_text("data test ")
    
    
        ax2 = fig.add_subplot(3,2,2)
        ax2.imshow(fake_images[:,:,1,1]|>cpu,vmin = 0,vmax = 1)
        ax2.title.set_text("digit pred 1 ")
    
    
        ax3 = fig.add_subplot(3,2,3)
        ax3.imshow(X[:,:,:,2],vmin = 0,vmax = 1)
        ax3.title.set_text("data test ")
    
    
        ax4 = fig.add_subplot(3,2,4)
        ax4.imshow(fake_images[:,:,1,2]|>cpu,vmin = 0,vmax = 1)
        ax4.title.set_text("digit pred 1 ")
    
    
    
        ax5 = fig.add_subplot(3,2,5)
        ax5.imshow(X[:,:,:,3],vmin = 0,vmax = 1)
        ax5.title.set_text("data test ")
    
    
        ax6 = fig.add_subplot(3,2,6)
        ax6.imshow(fake_images[:,:,1,3]|>cpu,vmin = 0,vmax = 1)
        ax6.title.set_text("digit pred 1 ")
    
    
        fig.savefig("number one test $e.png")
        plt.close(fig)
    

        ############# Training metric logs
          plt.plot(lossnrm)
          plt.title("loss $e")
          plt.savefig("lossnorm$e.png")
          plt.close()
          plt.plot(logdet_train)
          plt.title("logdet $e")
          plt.savefig("logdet$e.png")
          plt.close()
      
    end

    #save params every 4 epochs
    # if(mod(e,intermediate_save_params)==0) 
    #      # Saving parameters and logs
    #      Params = get_params(G) |> cpu 
    #      save_dict = @strdict e nepochs lr lr_step λ noiseLev n_hidden L K  Params floss flogdet num_digit total_seen
    #      @tagsave(
    #          datadir(sim_name, savename(save_dict, "jld2"; digits=6)),
    #          save_dict;
    #          safe=true
    #      )
    # end

end

