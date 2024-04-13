
using Plots

para = Array{Int32}(undef, 1, 3);
read!("param.bin", para);

TIME = para[1, 1]
NUM_SPATIAL = para[1, 2]
NUM_ELEMENTS = para[1, 3]


a = -1;
b = 1;
data = Array{Float64}(undef, 1, NUM_SPATIAL* NUM_ELEMENTS * TIME);
read!("meh.bin", data);

ele = NUM_SPATIAL* NUM_ELEMENTS

x = Array{Float64}(undef, NUM_ELEMENTS, NUM_SPATIAL)
read!("x.bin", x)
x = x';


@gif for i = 10000:TIME
    B = (reshape(data[1, (i-1)*ele +1 : (i)*ele], (NUM_ELEMENTS, NUM_SPATIAL)))';
    scatter(x, B, legend=false)

end

