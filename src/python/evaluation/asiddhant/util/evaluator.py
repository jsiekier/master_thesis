import os
import codecs
import torch
from .utils import *
import torch
from torch.autograd import Variable
import numpy as np
from sklearn import metrics
'''The code was provided by https://github.com/asiddhant/Active-NLP and adapted by Julia Siekiera'''
class Evaluator(object):
    def __init__(self, result_path, model_name, usecuda=True):
        self.result_path = result_path
        self.model_name = model_name
        self.usecuda = usecuda

    def evaluate(self, model, dataset, best_F, batch_size = 32):
        
        predicted_ids = []
        ground_truth_ids = []
        
        save = False
        new_F = 0.0
        
        data_batches = create_batches(dataset, batch_size = batch_size)

        for data in data_batches:

            words = data['words']

            if self.usecuda:
                words = Variable(torch.LongTensor(words)).cuda()
            else:
                words = Variable(torch.LongTensor(words))

            wordslen = data['wordslen']
            
            _, out = model.predict(words, wordslen, usecuda = self.usecuda)         
            
            ground_truth_ids.extend(data['tags'])
            predicted_ids.extend(out)

        new_F = np.mean(np.array(ground_truth_ids) == np.array(predicted_ids))
        #f1_measure = metrics.f1_score(ground_truth_ids, predicted_ids)
        if new_F > best_F:
            best_F = new_F
            save = True
        
        print('*'*80)
        print('Accuracy: %f, Best Accuracy: %f' %(new_F, best_F))
        print('*'*80)
            
        return best_F, new_F, save
        
        
    def evaluate_fmeasure(self, model, dataset, best_F, checkpoint_folder='.', batch_size = 32):
    
      predicted_ids = []
      ground_truth_ids = []
      
      save = False
      new_F = 0.0
      
      data_batches = create_batches(dataset, batch_size = batch_size)

      for data in data_batches:

          words = data['words']

          if self.usecuda:
              words = Variable(torch.LongTensor(words)).cuda()
          else:
              words = Variable(torch.LongTensor(words))

          wordslen = data['wordslen']
          
          _, out = model.predict(words, wordslen, usecuda = self.usecuda)         
          
          ground_truth_ids.extend(data['tags'])
          predicted_ids.extend(out)
          #print(data['tags'],out,'test')
      predicted_ids = np.array(predicted_ids)
      ground_truth_ids = np.array(ground_truth_ids)
      # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
      TP = np.sum(np.logical_and(predicted_ids == 1, ground_truth_ids == 1))
       
      # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
      TN = np.sum(np.logical_and(predicted_ids == 0, ground_truth_ids == 0))
       
      # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
      FP = np.sum(np.logical_and(predicted_ids == 1, ground_truth_ids == 0))
       
      # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
      FN = np.sum(np.logical_and(predicted_ids == 0, ground_truth_ids == 1))
       
      new_F = 2.0*TP/float((2*TP+FN+FP))
      print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))
      if new_F > best_F:
          best_F = new_F
          save = True
      
      print('*'*80)
      print('F-Measure: %f, Best F-Measure: %f' %(new_F, best_F))
      print('*'*80)
          
      return best_F, new_F, save
