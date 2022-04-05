lena = imread('lena.jpg');
%image(lena)

function result = tnorm(x)
    ave = sum(x(:))/numel(x); 
end