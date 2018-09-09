clear;
a = magic (3)
a(2, 3)
t = 0: 0.1: 10;
t = t';
y = sin(t);
b = inv (a);
c = pinv (a);
plot(t, y, 'rx')
d = a * b
e = a * c
a .* a
a .* 3
size(t,2)
length(a)