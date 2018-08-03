import numpy as np
import matplotlib.pyplot as plt


class ConfMatrix:
   """
   This class can build and display a confusion matrix.
   """

   def __init__(self, nclasses, labelset = None):
      self.nclasses = nclasses
      self.labelset = labelset
      self.cm = np.zeros((nclasses, nclasses))

   def add(self, gtlabels, estlabels):
      """
      This method adds data to the confusion matrix

      Takes
      gtlabels: array of ground truth labels
      estlabels: array of estiamated labels of SAME SIZE as gtlabels
      """

      if not len(gtlabels) == len(estlabels):
         raise Exception('intput gtlabels and estlabels must have the same length')
      for (gtl, estl) in zip(gtlabels, estlabels):
         self.cm[gtl, estl] += 1
      return self

   def sort(self, sort_index = None):
      if sort_index is None:
         totals = cm.sum(axis=1)
         sort_index = np.argsort(totals)[::-1]
      
      tmp = np.arange(self.cm.shape[0])
      cmperm = np.arange(self.cm.shape[0]);
      cmperm[sort_index] = tmp
      self.cm = self.collapse(cmperm)
      self.labelset = self.labelset[sort_index]
      return self

   def cut(self, newsize):
      cmperm = np.concatenate((np.arange(newsize, dtype=np.uint16), np.ones(self.nclasses - newsize, dtype=np.uint16) * newsize))
      self.cm = self.collapse(cmperm)
      self.labelset = self.labelset[:newsize]
      self.nclasses = newsize + 1
      self.labelset = np.concatenate((self.labelset[:newsize], np.asarray(['OTHER'])))
      return self



   def show(self, title='CM', collapsemap = None, cmap=plt.cm.Greys, normalize='recall', fontsize = 12, threshold = 0, title_with_acc = True):
      """
      This method plots the confusion matrix

      Takes
      title: default plot title
      labels: list of strings indicating the label names
      normalize: {'recall' or 'precision' or None}. 'recall' row-normalized the confusion matrix, 'precision' column-normalizes the conf. matrix, and None doesn't normalize.
      threshold: threshold above which to display the classification rates in the grid
      title_with_acc: {True, False}. If True, this appends accuracy and cohens kappa to the title.

      """
      cm = self.cm
      

      plt.rcParams.update({'font.size': fontsize})
      scale = 1
      nclasses = cm.shape[0]
      if normalize == 'recall':
         totals = cm.sum(axis=1)
         cm = cm / totals[:, np.newaxis]
         scale = 100
      elif normalize == 'precision':
         totals = cm.sum(axis=0)
         cm = cm / totals[np.newaxis, :]
         scale = 100
            
      plt.imshow(cm, interpolation='nearest', cmap=cmap)
      
      if(title_with_acc):
         (acc, cok) = self.get_accuracy()
         (cacc, ccok) = self.get_class_accuracy()
         recalls = self.get_class_recalls()
         precisions = self.get_class_precisions()
         f1s_denominator = precisions + recalls
         f1s_denominator[f1s_denominator == 0] = 1
         f1s = 2 * np.multiply(recalls, precisions) / f1s_denominator
         title = title + " [A:{:.1f}%, K:{:.1f}%, mA:{:.1f}%, mK:{:.1f}%, mR:{:.1f}%, mP:{:.1f}%, mF1:{:.1f}%]".format(100*acc, 100*cok, 100*np.mean(cacc), 100*np.mean(ccok), 100*np.mean(recalls), 100*np.mean(precisions), 100 * np.mean(f1s))
      plt.title(title)
      
      tick_marks = np.arange(nclasses)

      if self.labelset is not None:
         plt.xticks(tick_marks, self.labelset, rotation=45)
         plt.yticks(tick_marks, self.labelset)

      for x in range(nclasses):
         for y in range(nclasses):
            if abs(scale*cm[x][y])>threshold:
               value = "%2.0f%%" %(cm[x][y] * scale)
               plt.annotate(value, xy=(y, x), horizontalalignment='center', verticalalignment='center', color='blue', backgroundcolor = 'white')
            
            
      if normalize == 'recall':
         for y, val in enumerate(totals):
            plt.annotate("%.0f" %val, xy=(nclasses, y), horizontalalignment='center', verticalalignment='center', color='blue', backgroundcolor = 'white', annotation_clip=False)
      elif normalize == 'precision':
         for x, val in enumerate(totals):
            plt.annotate("%.0f" %val, xy=(x, nclasses-.65), horizontalalignment='center', verticalalignment='center', color='blue', backgroundcolor = 'white', annotation_clip=False)

                     
      plt.ylabel('True label')
      plt.xlabel('Predicted label')

   def get_class_accuracy(self, cm = None):

      if cm is None:
         cm = self.cm

      cok = np.zeros(self.nclasses)
      acc = np.zeros(self.nclasses)
      for i in range(self.nclasses):
         collapsemap = np.zeros(self.nclasses, dtype = np.uint8)
         collapsemap[i] = 1
         cmtemp = self.collapse(collapsemap)
         (acc[i], cok[i]) = self.get_accuracy(cm = cmtemp)
      return (acc, cok)


   def get_accuracy(self, cm = None):
      """
      This method calculates accuracy and Cohens Kappa from the confusion matrix
      """
      if cm is None:
         cm = self.cm
      
      acc = np.sum(np.diagonal(cm))/np.sum(cm)

      pgt = cm.sum(axis=1) / np.sum(cm) #probability of the ground truth to predict each class

      pest = cm.sum(axis=0) / np.sum(cm) #probability of the estimates to predict each class

      pe = np.sum(pgt * pest) #probaility of randomly guessing the same thing!

      if (pe == 1):
         cok = 1
      else:
         cok = (acc - pe) / (1 - pe) #cohens kappa!

      return (acc, cok)

   def collapse(self, collapsemap):

      cmin = self.cm
      nnew  = max(collapsemap) + 1
      cmint = np.zeros((self.nclasses, nnew)) #intermediate representation
      cmout = np.zeros((nnew, nnew))

      for i in range(nnew):
         cmint[:, i] = np.sum(cmin[:, collapsemap == i], axis = 1)

      for i in range(nnew):
         cmout[i, :] = np.sum(cmint[collapsemap == i, :], axis = 0)

      return cmout

   def get_class_recalls(self):
      cm = self.cm
      totals = cm.sum(axis=1)
      totals[totals == 0] = 1
      cm = cm / totals[:, np.newaxis]
      return(np.diag(cm))

   def get_class_precisions(self):
      cm = self.cm
      totals = cm.sum(axis = 0)
      totals[totals == 0] = 1
      cm = cm / totals[:, np.newaxis]
      return(np.diag(cm))

   def get_class_f1(self):
      recalls = self.get_class_recalls()
      precisions = self.get_class_precisions()
      f1s_denominator = precisions + recalls
      f1s_denominator[f1s_denominator == 0] = 1
      f1s = 2 * np.multiply(recalls, precisions) / f1s_denominator
      return f1s

   def export(self, normalize='recall'):

      cm = self.cm
      if normalize == 'recall':
         totals = cm.sum(axis=1)
         cm = cm / totals[:, np.newaxis]
         
      elif normalize == 'precision':
         totals = cm.sum(axis=0)
         cm = cm / totals[np.newaxis, :]
      return cm