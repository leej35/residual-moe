from torchtext import data, datasets
from TrainingUtils import *

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"

SRC = data.Field(pad_token=BLANK_WORD)
TGT = data.Field(init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD)

MAX_LEN = 50

train, val, test = datasets.IWSLT.splits(
    exts=('.de', '.en'), fields=(SRC, TGT), 
    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
        len(vars(x)['trg']) <= MAX_LEN)

# train, val, test = data.TabularDataset.splits(path="", train="train/data.train.tsv", validation="validation/data.val.tsv", test="test/data.test.tsv",
#                                               format='tsv', fields=[('src', SRC), ('trg', TGT)],
#                                               filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)

MIN_FREQ = 2
SRC.build_vocab(train, min_freq=MIN_FREQ, max_size=40000)
TGT.build_vocab(train, min_freq=MIN_FREQ, max_size=40000)


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


pad_idx = TGT.vocab.stoi[BLANK_WORD]
model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
model.cuda()
criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
criterion.cuda()
BATCH_SIZE = 64
# train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)
# valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)
train_iter = data.BucketIterator(train, batch_size=BATCH_SIZE, device=0, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), train=True)
valid_iter = data.BucketIterator(val, batch_size=BATCH_SIZE, device=0, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), train=False)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()
    run_epoch((rebatch(pad_idx, b) for b in train_iter), model, SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model, SimpleLossCompute(model.generator, criterion, None))
    print(loss)
