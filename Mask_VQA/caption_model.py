import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from data_loacer_for_only_caption import get_loader

class EncoderCNN(nn.Module):
    def __init__(self,embed_size):
        super(EncoderCNN, self).__init__()
        self.inception = models.inception_v3(pretrained=True,aux_logits=False)
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)

        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True

        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(DecoderRNN,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,num_layers)
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self,features,captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0),embeddings),dim = 0)
        hiddens,_ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(CNNtoRNN,self).__init__()
        self.encoderCNN =EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size,hidden_size,vocab_size,num_layers)

    def forward(self,image,captions):
        features = self.encoderCNN(image)
        outputs = self.decoderRNN(features,captions)
        return outputs

    def caption_images(self,image,vocabulary,max_length= 50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens,states = self.decoderRNN.lstm(x,states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)

                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] =="<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]

if __name__ == "__main__":

    input_dir = 'D:/data/vqa/coco/simple_vqa'
    log_dir = './logs'
    model_dir = './models'
    max_qst_length = 30
    max_cap_length = 50
    max_num_ans = 10
    embed_size = 64
    word_embed_size = 300
    num_layers = 2
    hidden_size = 16
    learning_rate = 0.001
    step_size = 10
    gamma = 0.1
    num_epochs = 50
    batch_size = 64
    num_workers = 0
    save_step = 1
    step = 0

    train_loader = get_loader(
        input_dir=input_dir,
        input_vqa_train='train.npy',
        input_vqa_valid='valid.npy',
        max_qst_length=max_qst_length,
        max_cap_length=max_cap_length,
        max_num_ans=max_num_ans,
        batch_size=batch_size,
        num_workers=num_workers)

    cap_vocab_size = train_loader['train'].dataset.cap_vocab.vocab_size

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    # initialize model , loss etc
    model = CNNtoRNN(embed_size, hidden_size, cap_vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    model.train()

    for epoch in range(num_epochs):

        for phase in ['train']:
            for idx, batch_sample in enumerate(train_loader[phase]):
                imgs = batch_sample['image'].to(device).float()
                captions = batch_sample['caption'].to(device)

                outputs = model(imgs, captions)
                loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

                print("Training loss", loss.item())
                step += 1

                optimizer.zero_grad()
                loss.backward(loss)
                optimizer.step()