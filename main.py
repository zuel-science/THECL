import warnings
warnings.filterwarnings("ignore")

import argparse
import os.path as osp
from pathlib import Path
import shutil

import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split

import torch
import torch_geometric.transforms as T

from model import THECL, Encoder
from utils import set_random_seed, add_self_loop
from datasets import get_dataset

def get_arguments():
    parser = argparse.ArgumentParser(description='Training model')
    parser.add_argument('--dataset', type=str, default='acm')
    parser.add_argument("--train_target", type=str, choices=["micro_f1", "nmi"], default="micro_f1", 
                        help="Training target to optimize ('micro_f1' or 'nmi')")

    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--clf_runs', type=int, default=10)

    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--edge_drop_rate', type=float, default=0.2)
    parser.add_argument('--feature_drop_rate', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--moving_average_decay', type=float, default=0.)

    parser.add_argument('--train_splits', type=float, nargs='+', default=[0.2])
    parser.add_argument('--combine', type=str, default='concat')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    return vars(args)


def train(model, x, edge_indices, optimizer):
    model.train()
    optimizer.zero_grad()

    loss = model.loss(x, edge_indices)

    loss.backward()
    optimizer.step()
    model.update_ma()

    return loss.item()


def test(embeddings, labels, train_split=0.2, runs=10):
    macro_f1_list = list()
    micro_f1_list = list()
    nmi_list = list()
    ari_list = list()

    for i in range(runs):
        x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, train_size=train_split, random_state=i)

        clf = SVC(probability=True)

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        macro_f1 = f1_score(y_test, y_pred, average='macro')
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1_list.append(macro_f1)
        micro_f1_list.append(micro_f1)

    for i in range(runs):
        kmeans = KMeans(n_clusters=len(torch.unique(labels)), algorithm='lloyd')
        y_kmeans = kmeans.fit_predict(embeddings)

        nmi = normalized_mutual_info_score(labels, y_kmeans)
        ari = adjusted_rand_score(labels, y_kmeans)
        nmi_list.append(nmi)
        ari_list.append(ari)

    macro_f1 = np.array(macro_f1_list).mean()
    micro_f1 = np.array(micro_f1_list).mean()
    nmi = np.array(nmi_list).mean()
    ari = np.array(ari_list).mean()

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1, 
        'nmi': nmi,
        'ari': ari
    }



def main():
    params = get_arguments()
    set_random_seed(params['seed'])
    train_target = params['train_target']
    device = torch.device('cuda:{}'.format(params['gpu']) if torch.cuda.is_available() else 'cpu')

    checkpoints_path = f'checkpoints'
    try:
        shutil.rmtree(checkpoints_path)
    except:
        pass
    Path(checkpoints_path).mkdir(parents=True, exist_ok=False)

    dataset, metapaths, target = get_dataset(params['dataset'])
    data = dataset[0]
    num_relations = len(metapaths)
    num_nodes = data[target].y.shape[0]
    num_feat = data[target].x.shape[1]

    metapath_data = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True)(data)
    metapath_data = add_self_loop(metapath_data, num_relations, num_nodes)

    x = metapath_data[target].x.to(device)
    edge_indices = [edge_index.to(device) for edge_index in metapath_data.edge_index_dict.values()]
    labels = metapath_data[target].y

    encoder = Encoder(in_dim=num_feat, hid_dim=params['hid_dim'], num_layers=params['num_layers'])                 
    model = THECL(encoder=encoder, hid_dim=params['hid_dim'], num_relations=num_relations,
                 tau=params['tau'], pe=params['edge_drop_rate'], pf=params['feature_drop_rate'],
                 alpha=params['alpha'], moving_average_decay=params['moving_average_decay']).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    model = model

    best_epoch = 0
    best_value = 0
    patience_cnt = 0

    for i in range(1, 3000):
        loss = train(model, x, edge_indices, optimizer)

        if i % 10 == 0:
            embeddings = model(x, edge_indices, 'concat').detach().cpu().numpy()
            results = test(embeddings, labels, train_split=0.2, runs=params['clf_runs'])
            print(f'Macro-F1: {results["macro_f1"]:.4f} | Micro-F1: {results["micro_f1"]:.4f} | NMI: {results["nmi"]:.4f} | ARI: {results["ari"]:.4f}')

            if train_target == 'micro_f1':
                if results['micro_f1'] > best_value:
                    best_value = results['micro_f1']
                    best_epoch = i
                    patience_cnt = 0
                    torch.save(model.state_dict(), osp.join(checkpoints_path, f'{i}.pkl'))
                else:
                    patience_cnt += 1
            elif train_target == 'nmi':
                if results['nmi'] > best_value:
                    best_value = results['nmi']
                    best_epoch = i
                    patience_cnt = 0
                    torch.save(model.state_dict(), osp.join(checkpoints_path, f'{i}.pkl'))
                else:
                    patience_cnt += 1

            if patience_cnt == 20:
                break

    model.load_state_dict(torch.load(osp.join(checkpoints_path, f'{best_epoch}.pkl')))
    shutil.rmtree(checkpoints_path)

    embeddings = model(x, edge_indices,'concat').detach().cpu().numpy()
    labels = metapath_data[target].y
    
    results = test(embeddings, labels, train_split=0.2, runs=params['clf_runs'])

    print(f'Train Split: 0.2 | Macro-F1: {results["macro_f1"]:.4f} | Micro-F1: {results["micro_f1"]:.4f} | NMI: {results["nmi"]:.4f} | ARI: {results["ari"]:.4f}')


if __name__ == '__main__':
    main()