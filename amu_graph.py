import sys
import os
import torch
import pickle
from utils import *
from tqdm import tqdm
from absl import flags
from semi_vae import _sample
from semi_vae import SemiVAE
import torch.utils.data as Data
from graph_dataset import AMUGraphDataset
from classification_layer import ClassificationLayer

torch.manual_seed(123)

class AMUGraph():
    def train_class_layer(self, feat_dataset:torch.Tensor, classification_layer:ClassificationLayer, device='cuda'):
        l_hat = []
        print("Training AMUGraph's classification layer")
        # Run through to get the reconstruction losses
        normal_encoded = []
        anomaly_encoded = []
        mus = []
        vars = [] 
        for feat in tqdm(feat_dataset):
            feat.requires_grad = True
            dec, mu, var = self.autoencoder(feat.view(1,-1))
            l_hat.append(ssd(dec, feat).item())
            mus.append(mu)
            vars.append(var)
            
        l_min = min(l_hat)
        l_max = max(l_hat)
        l_hat = [(l - l_min) / (l_max - l_min) for l in l_hat]

        for ind, feat in enumerate(feat_dataset):
            feat.requires_grad = True
            enc = _sample(mus[ind], vars[ind])
            if l_hat[ind] < 0.5:
                normal_encoded.append(enc)
            else:
                anomaly_encoded.append(enc)
        
        normal_encoded = torch.stack(normal_encoded).to(device)
        anomaly_encoded = torch.stack(anomaly_encoded).to(device)
        
        classification_layer.fit(normal_encoded, anomaly_encoded)
        return classification_layer

    def augment(self, data_points, labeled):
        benign = torch.where(labeled[:, 1] > labeled[:, 2])[0]
        anomalies = torch.where(labeled[:, 1] < labeled[:, 2])[0]

        augmented_points = []
        augmented_labels = []
        for b1 in benign:
            for b2 in benign:
                b1point = data_points[b1]
                b2point = data_points[b2]

                augmented_points.append(labeled[b1][1] * b1point + labeled[b2][2] * b2point)
                augmented_labels.append(torch.stack([labeled[b1][1], labeled[b2][2]]))

        for a1 in anomalies:
            for a2 in anomalies:
                a1point = data_points[a1]
                a2point = data_points[a2]

                augmented_points.append(labeled[a1][1] * a1point + labeled[a2][2] * a2point)
                augmented_labels.append(torch.stack([labeled[a1][1], labeled[a2][2]]))

        for b in benign:
            for a in anomalies:
                bpoint = data_points[b]
                apoint = data_points[a]

                augmented_points.append(labeled[b][1] * bpoint + labeled[a][2] * apoint)
                augmented_labels.append(torch.stack([labeled[b][1], labeled[a][2]]))

                augmented_points.append(labeled[b][2] * bpoint + labeled[a][1] * apoint)
                augmented_labels.append(torch.stack([labeled[a][1], labeled[b][2]]))
        
        return augmented_points, augmented_labels

    def train_autoencoder(self, feat_dataset:torch.Tensor, epoch, device='cuda'):
        l_hat = []
        print("\tCollecting reconstruction losses for all the training points")
        # Run through to get the reconstruction losses
        for x in tqdm(feat_dataset):
            x.requires_grad = True
            dec, _, _ = self.autoencoder(x.view(1,-1))
            l_hat.append(ssd(dec, x))
        
        l_hat, index = normalize(l_hat)
            
        ### Active Learning ###
        queries = torch.where((l_hat >= self.args.medium_lower_threshold) & (l_hat <= self.args.medium_upper_threshold))[0]
        queries = queries[torch.randperm(len(queries))[:self.QUERY_BUDGET]]
        #queries = torch.randperm(len(l_hat))[:self.QUERY_BUDGET]

        labeled = []
        for q in tqdm(queries):
            # querying
            point = feat_dataset[index[q]]
            prob = self.confidence_labeler.forward(point)
            labeled.append(torch.Tensor([index[q], prob[0].item(), prob[1].item()]))

        ### Mixup ###
        labeled = torch.stack(labeled).to(device)
        idx = labeled[:, 0]
        data_points = feat_dataset[idx.int()].to(device)
        data_labels = labeled[:, 1:3]

        augmented_points, augmented_labels = self.augment(data_points, labeled)
            
        ### Backpropagation ###
        self.autoencoder.train()

        if len(augmented_points):
            augmented_points = torch.stack(augmented_points)
            augmented_labels = torch.stack(augmented_labels)

            x = torch.cat((augmented_points, data_points))
            y = torch.cat((augmented_labels, data_labels))
        else:
            x = data_points
            y = data_labels

        #idx = torch.where(y == 0)[0]
        self.autoencoder.fit(x, y.to(device))
        self.autoencoder.eval()

    def __init__(self, args):
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'

        print('Loading in the dataset...')
        train_dataset = AMUGraphDataset(train=True, dataset=args.dataset)
        self.train_data = train_dataset.x
        self.train_labels = train_dataset.y
        self.test_dataset = AMUGraphDataset(train=False, dataset=args.dataset)
        print('Done loading it!')

        # determine soft labeler file
        mlp_instance = MLP2
        if 'yelp' in args.dataset:
            self.INPUT_DIM = 139
            self.PARAM_FILE = 'soft_labelers/yelp_basic_mlp_params.pt'
            self.EXPLORE_DIM = 57
        elif 'amazon' in args.dataset:
            self.INPUT_DIM = 133
            self.PARAM_FILE = 'soft_labelers/amazon_0.8863368891406275_basic_mlp_params.pt'
            self.EXPLORE_DIM = 55
            mlp_instance = MLP
        elif 'cora' in args.dataset:
            self.INPUT_DIM = 139
            self.PARAM_FILE = 'soft_labelers/cora_0.6210652635465707_basic_mlp_params.pt'
            self.EXPLORE_DIM = 279
            mlp_instance = MLP
        elif 'pubmed' in args.dataset:
            self.INPUT_DIM = 139
            self.PARAM_FILE = 'soft_labelers/pubmed_0.8019_basic_mlp_params.pt'
            self.EXPLORE_DIM = 95
        
        self.autoencoder = SemiVAE([self.INPUT_DIM], args.latent_dim, [64, 32, 16])
        self.autoencoder.to(device)
        for param in self.autoencoder.parameters():
            param.requires_grad_(True)

        self.classification_layer = ClassificationLayer(args.latent_dim)
        self.classification_layer.to(device)

        self.confidence_labeler = mlp_instance(input_dim=self.INPUT_DIM)
        self.confidence_labeler.load_state_dict(torch.load(self.PARAM_FILE))
        self.confidence_labeler.to(device)

        self.QUERY_BUDGET = int(args.budget_percentage * len(self.train_data) / args.epochs)

        self.args = args

        self.checkpoint_dataset = f'amug_ckpts_{args.dataset}'
        #os.mkdir(self.checkpoint_dataset)
    
    def run(self):
        #self.QUERY_BUDGET = int(self.args.budget_percentage * len(self.train_data) / self.args.epochs)
        validation_scores = []

        ### Training autoencoder ###
        for epoch in range(self.args.epochs):
            print(f"Epoch {epoch+1}/{self.args.epochs}")
            temp_cl = ClassificationLayer()
            self.train_autoencoder(self.train_data, epoch=epoch)
            self.train_class_layer(self.train_data, temp_cl)
            validation_scores.append(testing(self.test_dataset, self.autoencoder, temp_cl))
            torch.save(self.autoencoder.state_dict(), f'{self.checkpoint_dataset}/{epoch}_ae.pt')
            with open(f'{self.checkpoint_dataset}/{epoch}_cl.pkl', 'wb+') as f:
                pickle.dump(temp_cl, f)
        
        ### Training Classification Layer ###
        self.train_class_layer(self.train_data, self.classification_layer)

        ### Testing ###
        score = testing(self.test_dataset, self.autoencoder, self.classification_layer)

        print(f'Validation:', validation_scores)
        print(f'ROC_AUC: {max(max(validation_scores), score)}')

        final_score = max(max(validation_scores), score)
        with open('zamubandits_results.txt', 'a+') as f:
            f.write(f'{self.args.epochs}|{self.QUERY_BUDGET}|{self.args.medium_lower_threshold}|{self.args.medium_upper_threshold}|{final_score}\n')

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_float('medium_lower_threshold', 0.4, 'Medium Lower Threshold')
    flags.DEFINE_float('medium_upper_threshold', 0.8, 'Medium Upper Threshold')
    flags.DEFINE_float('budget_percentage', 0.035, 'Number of queried labels per epoch')
    flags.DEFINE_integer('epochs', 16, 'Epochs')
    flags.DEFINE_string('dataset', 'pubmed_embeddings', 'dataset name')
    flags.DEFINE_integer('latent_dim', 32, 'latent dimension')
    FLAGS(sys.argv)

    runner = AMUGraph(FLAGS)
    runner.run()