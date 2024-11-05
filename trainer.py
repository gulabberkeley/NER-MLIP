import torch
import torch.nn as nn
import utils as ut
import matplotlib.pyplot as plt
from datetime import date
import os
import numpy as np
import pandas as pd
import json

def train(model, tokenizer, record_list_train, record_list_test, record_list_ood_1, record_list_ood_2, classes, 
          batch_size, seed, max_length, class_weights, lr, n_epochs, linear_probe=False,
          plot=True, save_model=True, save_results=True):
    folder = f'{date.today()}_n_{len(record_list_train)}_l_{lr}_lp_{linear_probe}_w_{class_weights}_b_{batch_size}_s_{seed}'
    os.makedirs(f'saved_models/{folder}', exist_ok=True)
    data_batches, target_batches, att_mask_batches = ut.preprocess(record_list=record_list_train, classes=classes, tokenizer=tokenizer, 
                                                                   batch_size=batch_size, max_length=max_length, test=False)
    weights = torch.tensor(class_weights)
    weights_n = weights / torch.norm(weights)
    weights_n = torch.cat((weights_n, torch.tensor([0])))  # weights for padding = 0
    criterion = nn.CrossEntropyLoss(weight=weights_n)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if linear_probe:
        # Freeze weights of all but the last layer
        for param in model.bert.parameters():
            param.requires_grad = False

    train_losses = []
    train_precisions, train_recalls, train_f1s = [], [], []
    test_precisions, test_recalls, test_f1s = [], [], []
    ood_1_precisions, ood_1_recalls, ood_1_f1s = [], [], []
    ood_2_precisions, ood_2_recalls, ood_2_f1s = [], [], []

    for epoch in range(n_epochs):
        epoch += 1
        train_loss_batch = []
        train_precision_batch, train_recall_batch, train_f1_batch = [], [], []
        for b, X in enumerate(data_batches):
            y_pred = model(X, attention_mask=att_mask_batches[b])
            y_pred = torch.swapaxes(y_pred, 1, 2)
            y = target_batches[b]            
            loss = criterion(y_pred, y)
            precision, recall, f1 = ut.scores(len(classes), y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_batch.append(loss.item())
            train_precision_batch.append(precision)
            train_recall_batch.append(recall)
            train_f1_batch.append(f1)
        
        train_loss_batch_mean = ut.calc_mean(train_loss_batch)
        train_precision_batch_mean = ut.calc_mean(train_precision_batch)
        train_recall_batch_mean = ut.calc_mean(train_recall_batch)
        train_f1_batch_mean = ut.calc_mean(train_f1_batch)
        print(f'Epoch {epoch}')    
        print(f'Mean training precision: {train_precision_batch_mean:.4f}')
        print(f'Mean training recall: {train_recall_batch_mean:.4f}')
        print(f'Mean training f1: {train_f1_batch_mean:.4f}')
        train_losses.append(train_loss_batch_mean)
        train_precisions.append(train_precision_batch_mean)
        train_recalls.append(train_recall_batch_mean)
        train_f1s.append(train_f1_batch_mean)

        precision_test, recall_test, f1_test, pred_test, true_test = testing(model, record_list_test, classes, tokenizer, max_length)
        print(f'Mean test precision: {precision_test:.4f}')
        print(f'Mean test recall: {recall_test:.4f}')
        print(f'Mean test f1: {f1_test:.4f}')
        test_precisions.append(precision_test)
        test_recalls.append(recall_test)
        test_f1s.append(f1_test)
            
        precision_ood_1, recall_ood_1, f1_ood_1, pred_ood_1, true_ood_1 = testing(model, record_list_ood_1, classes, tokenizer, max_length)
        print(f'Mean test_ood_1 precision: {precision_ood_1:.4f}')
        print(f'Mean test_ood_1 recall: {recall_ood_1:.4f}')
        print(f'Mean test_ood_1 f1: {f1_ood_1:.4f}')
        ood_1_precisions.append(precision_ood_1)
        ood_1_recalls.append(recall_ood_1)
        ood_1_f1s.append(f1_ood_1)

        precision_ood_2, recall_ood_2, f1_ood_2, pred_ood_2, true_ood_2 = testing(model, record_list_ood_2, classes, tokenizer, max_length)
        print(f'Mean test_ood_2 precision: {precision_ood_2:.4f}')
        print(f'Mean test_ood_2 recall: {recall_ood_2:.4f}')
        print(f'Mean test_ood_2 f1: {f1_ood_2:.4f}')
        ood_2_precisions.append(precision_ood_2)
        ood_2_recalls.append(recall_ood_2)
        ood_2_f1s.append(f1_ood_2)
            
        # Save model and results    
        model_name = f'{folder}/{folder}_e_{epoch}'
        if save_model:             
            torch.save(model.state_dict(), f"saved_models/{model_name}.pt")
        ut.save_annotations(record_list_test, true_test, pred_test, model_name, 'test')
        ut.save_annotations(record_list_ood_1, true_ood_1, pred_ood_1, model_name, 'ood_1')
        ut.save_annotations(record_list_ood_2, true_ood_2, pred_ood_2, model_name, 'ood_2')

    if save_results:
        train_losses_np = np.array(train_losses)
        train_precisions_np = np.array(train_precisions)
        train_recalls_np = np.array(train_recalls)
        train_f1s_np = np.array(train_f1s)
        
        test_precisions_np = np.array(test_precisions)
        test_recalls_np = np.array(test_recalls)
        test_f1s_np = np.array(test_f1s)
        
        ood_1_precisions_np = np.array(ood_1_precisions)
        ood_1_recalls_np = np.array(ood_1_recalls)
        ood_1_f1s_np = np.array(ood_1_f1s)

        ood_2_precisions_np = np.array(ood_2_precisions)
        ood_2_recalls_np = np.array(ood_2_recalls)
        ood_2_f1s_np = np.array(ood_2_f1s)

        data = np.vstack((train_losses_np, train_precisions_np, train_recalls_np, train_f1s_np,
                          test_precisions_np, test_recalls_np, test_f1s_np,
                          ood_1_precisions_np, ood_1_recalls_np, ood_1_f1s_np,
                          ood_2_precisions_np, ood_2_recalls_np, ood_2_f1s_np)).T
        data_df = pd.DataFrame(data, columns=['train_loss', 'train_precision', 'train_recall', 'train_f1',
                                'test_precision', 'test_recall', 'test_f1',
                                'ood_1_precision', 'ood_1_recall', 'ood_1_f1',
                                'ood_2_precision', 'ood_2_recall', 'ood_2_f1'])
        data_df.to_csv(f'saved_models/{folder}/results.csv', index=False)
    
    if plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(4, 8))
        fig.tight_layout()

        ax1.plot(range(1, n_epochs + 1), train_losses, 'o-', c='red', label='training')
        ax1.legend(fontsize=12)
        ax1.set_ylabel("loss", fontsize=12)

        ax2.plot(range(1, n_epochs + 1), train_precisions, 'o-', c='blue', label='training')
        ax2.plot(range(1, n_epochs + 1), test_precisions, 'o-', c='green', label='test')
        ax2.plot(range(1, n_epochs + 1), ood_1_precisions, 'o-', c='magenta', label='OOD_1')
        ax2.plot(range(1, n_epochs + 1), ood_2_precisions, 'o-', c='orange', label='OOD_2')
        ax2.legend(fontsize=12)
        ax2.set_ylabel("precision", fontsize=12)

        ax3.plot(range(1, n_epochs + 1), train_recalls, 'o-', c='blue', label='training')
        ax3.plot(range(1, n_epochs + 1), test_recalls, 'o-', c='green', label='test')
        ax3.plot(range(1, n_epochs + 1), ood_1_recalls, 'o-', c='magenta', label='OOD_1')
        ax3.plot(range(1, n_epochs + 1), ood_2_recalls, 'o-', c='orange', label='OOD_2')
        ax3.legend(fontsize=12)
        ax3.set_ylabel("recall", fontsize=12)

        ax4.plot(range(1, n_epochs + 1), train_f1s, 'o-', c='blue', label='training')
        ax4.plot(range(1, n_epochs + 1), test_f1s, 'o-', c='green', label='test')
        ax4.plot(range(1, n_epochs + 1), ood_1_f1s, 'o-', c='magenta', label='OOD_1')
        ax4.plot(range(1, n_epochs + 1), ood_2_f1s, 'o-', c='orange', label='OOD_2')
        ax4.legend(fontsize=12)
        ax4.set_xlabel("epoch", fontsize=12)
        ax4.set_ylabel("F1 score", fontsize=12)

        plt.show()

    return pred_test, pred_ood_1, pred_ood_2


def testing(model, record_list, classes, tokenizer, max_length):
    data_test, target_test, att_mask_test = ut.preprocess(record_list, classes, tokenizer, 
                                                          batch_size=0, max_length=max_length, test=True)
    with torch.no_grad():
        y_pred_test = model(data_test, attention_mask=att_mask_test)
        y_pred_test = torch.swapaxes(y_pred_test, 1, 2)
        precision_test, recall_test, f1_test = ut.scores(len(classes), y_pred_test, target_test)
    return precision_test, recall_test, f1_test, y_pred_test, target_test