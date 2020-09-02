import re
import numpy as np
import matplotlib.pyplot as plt

no_decimal = re.compile(r'[^\d]+')

trans = []
year = []
for line in open('moore.csv'):
    r = line.split('\t')
    x = int(no_decimal.sub('', r[1].split('[')[0]))
    y = int(r[2].split('[')[0])
    trans.append(x)
    year.append(y)

# convert data to array
year = np.array(year)
trans = np.array(trans)

# # plot raw data
# plt.scatter(year, trans, c='black', marker='2')
# plt.show()

# calculate log of trans
Y = np.log(trans)

# # plot transformed data
# plt.scatter(year, Y, c='black', marker='2')
# plt.show()

a = (year.dot(Y) - Y.mean() * sum(year)) / (year.dot(year) - year.mean() * sum(year))
b = Y.mean() - a * year.mean()
Y_hat = a * year + b
del1 = Y - Y.mean()
del2 = Y - Y_hat
R2 = 1 - del2.dot(del2) / del1.dot(del1)

plt.scatter(year, Y, c='black', marker='2')
plt.plot(year, Y_hat)
plt.show()

print(f'a = {a}')
print(f'b = {b}')
print(f'R² = {R2}')

# log(tc) = a.year + b
# tc = 10^(a.year + b) = 10^(a.year) * 10^b
# doubling the transistors: 2 * tc = 10^(log(2)) * 10^(a.year) * 10^b = 10^(a.year+ log(2)) * 10^b
# left side = tc2 = 10^(a.year2) * 10^b
# therefore: 10^(a.year2) * 10^b = 10^(a.year1 + log(2)) * 10^b
# year2 = year1 + log(2)/a

print(f'Transistors double every {np.log(2)/a} years!')
