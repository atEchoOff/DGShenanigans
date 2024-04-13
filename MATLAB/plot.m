a = -1;
b = 1;
f = fopen("meh.bin");
A = fread(f, 'double');

f = fopen("param.bin");
para = fread(f, 'int');

TIME = para(1);
NUM_SPATIAL = para(2);
NUM_ELEMENTS = para(3);
ele = NUM_SPATIAL* NUM_ELEMENTS;
B = zeros(TIME, NUM_SPATIAL, NUM_ELEMENTS);

f = fopen("x.bin");
x = fread(f, [NUM_ELEMENTS NUM_SPATIAL], 'double')';

for i = 1:TIME
    B = reshape(A((i-1)*ele +1 : (i)*ele), [NUM_ELEMENTS NUM_SPATIAL])';
    scatter(x, B, "filled");
    axis([a b 0 1.5]);
    drawnow limitrate
end


