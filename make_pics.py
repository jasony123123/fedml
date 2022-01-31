# import matplotlib.pyplot as plt

# f = open("diagram_inputs", "r")
# fdml = f.readlines()
# fdml = fdml[5:]
# data = {
#     'round': [],
#     'train_acc': [],
#     'train_loss': [],
#     'test_acc': [],
#     'test_loss': []}
# for i in range(len(fdml)):
#     fdml[i] = fdml[i].strip(' \nabcdefghijklmnopqrstuvwxyz')
#     fdml[i] = float(fdml[i])
#     md = i % 5
#     if md == 0:
#         print('round', int(fdml[i]))
#     elif md == 1:
#         data['train_acc'].append(fdml[i])
#     elif md == 2:
#         data['train_loss'].append(fdml[i])
#     elif md == 3:
#         data['test_acc'].append(fdml[i])
#     elif md == 4:
#         data['test_loss'].append(fdml[i])

# plt.plot(data['train_acc'], label='train_acc')
# plt.plot(data['test_acc'], label='test_acc')
# plt.legend()
# plt.show()

# plt.plot(data['train_loss'], label='train_loss')
# plt.plot(data['test_loss'], label='test_loss')
# plt.legend()
# plt.show()

# f.close()

import matplotlib.pyplot as plt

f = open("diagram_inputs2", "r")
fdml = f.readlines()
fdml = fdml[1:-2]
print(fdml)
data = {
    'round': [],
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': []}
for i in range(len(fdml)):
    fdml[i] = fdml[i].strip(' \nabcdefghijklmnopqrstuvwxyz')
    fdml[i] = float(fdml[i])
    md = i % 5
    if md == 0:
        print('round', int(fdml[i]))
    elif md == 1:
        data['train_loss'].append(fdml[i])
    elif md == 2:
        data['train_acc'].append(fdml[i])
    elif md == 3:
        data['test_loss'].append(fdml[i])
    elif md == 4:
        data['test_acc'].append(fdml[i])

plt.plot(data['train_acc'], label='train_acc')
plt.plot(data['test_acc'], label='test_acc')
plt.legend()
plt.show()

plt.plot(data['train_loss'], label='train_loss')
plt.plot(data['test_loss'], label='test_loss')
plt.legend()
plt.show()

# f.close()
