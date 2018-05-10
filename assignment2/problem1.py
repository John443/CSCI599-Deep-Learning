from lib.rnn import *
from lib.layer_utils import *
from lib.grad_check import *
from lib.optim import *
from lib.train import *
import numpy as np
import re
# import matplotlib.pyplot as plt

input_text = "Dog goes woof, cat goes meow. Bird goes tweet, and mouse goes squeak. Cow goes moo. Frog goes croak, and the elephant goes toot. Ducks say quack and fish go blub, and the seal goes ow ow ow. But there's one sound that no one knows... What does the fox say? Ring-ding-ding-ding-dingeringeding! Gering-ding-ding-ding-dingeringeding! Gering-ding-ding-ding-dingeringeding! What the fox say? Wa-pa-pa-pa-pa-pa-pow! Wa-pa-pa-pa-pa-pa-pow! Wa-pa-pa-pa-pa-pa-pow! What the fox say? Hatee-hatee-hatee-ho! Hatee-hatee-hatee-ho! Hatee-hatee-hatee-ho! What the fox say? Joff-tchoff-tchoff-tchoffo-tchoffo-tchoff! Joff-tchoff-tchoff-tchoffo-tchoffo-tchoff! Joff-tchoff-tchoff-tchoffo-tchoffo-tchoff! What the fox say? Big blue eyes, pointy nose, chasing mice, and digging holes. Tiny paws, up the hill, suddenly you're standing still. Your fur is red, so beautiful, like an angel in disguise. But if you meet a friendly horse, will you communicate by mo-o-o-o-orse, mo-o-o-o-orse, mo-o-o-o-orse? How will you speak to that h-o-o-orse, h-o-o-orse, h-o-o-orse? What does the fox say?! Jacha-chacha-chacha-chow! Jacha-chacha-chacha-chow! Jacha-chacha-chacha-chow! What the fox say? Fraka-kaka-kaka-kaka-kow! Fraka-kaka-kaka-kaka-kow! Fraka-kaka-kaka-kaka-kow! What the fox say? A-hee-ahee ha-hee! A-hee-ahee ha-hee! A-hee-ahee ha-hee! What the fox say? A-oo-oo-oo-ooo! Woo-oo-oo-ooo! What does the fox say?! The secret of the fox, ancient mystery. Somewhere deep in the woods, I know you're hiding. What is your sound? Will we ever know? Will always be a mystery what do you say? You're my guardian angel hiding in the woods. What is your sound? A-bubu-duh-bubu-dwee-dum a-bubu-duh-bubu-dwee-dum Will we ever know? A-bubu-duh-bubu-dwee-dum I want to, I want to, I want to know! A-bubu-duh-bubu-dwee-dum Bay-buh-day bum-bum bay-dum"

text = re.split(' |\n',input_text.lower()) # all words are converted into lower case
outputSize = len(text)
word_list = list(set(text))
dataSize = len(word_list)
output = np.zeros(outputSize)
for i in range(0, outputSize):
    index = np.where(np.asarray(word_list) == text[i])
    output[i] = index[0]
data_labels = output.astype(np.int)
gt_labels = data_labels[1:]
data_labels = data_labels[:-1]

print('Input text size: %s' % outputSize)
print('Input word number: %s' % dataSize)

# you can change the following parameters.
D = 10 # input dimention
H = 20 # hidden space dimention
T = 50 # timesteps
N = 10 # batch size
max_epoch = 200 # max epoch size

loss_func = temporal_softmax_loss()
# you can change the cell_type between 'rnn' and 'lstm'.
model = LanguageModelRNN(dataSize, D, H, cell_type='lstm')
optimizer = Adam(model, 5e-4)

data = { 'data_train': data_labels, 'labels_train': gt_labels }

results = train_net(data, model, loss_func, optimizer, timesteps=T, batch_size=N, max_epochs=max_epoch, verbose=True)
