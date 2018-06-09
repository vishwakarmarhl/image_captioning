import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from collections import defaultdict

import skimage.io as io
import skimage

import six.moves.cPickle as pickle
from utils.coco.coco import COCO
from utils.coco.pycocoevalcap.eval import COCOEvalCap
from utils.misc import CaptionData, ImageLoader, TopN
from utils.nn import NN

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.train_cnn = self.is_train and config.train_cnn
        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')
        self.image_shape = [224, 224, 3]
        self.nn = NN(config)
        self.global_step = tf.Variable(0,
                                       name = 'global_step',
                                       trainable = False)
        self.build()

    def build(self):
        raise NotImplementedError()

    def train(self, sess, train_data):
        """ Train the model using the COCO train2014 data. """
        print("Training the model...")
        config = self.config

        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)

        for _ in tqdm(list(range(config.num_epochs)), desc='epoch'):
            for _ in tqdm(list(range(train_data.num_batches)), desc='batch'):
                batch = train_data.next_batch()
                image_files, sentences, masks = batch
                images = self.image_loader.load_images(image_files)
                feed_dict = {self.images: images,
                             self.sentences: sentences,
                             self.masks: masks}
                _, summary, global_step = sess.run([self.opt_op,
                                                    self.summary,
                                                    self.global_step],
                                                    feed_dict=feed_dict)
                if (global_step + 1) % config.save_period == 0:
                    self.save()
                train_writer.add_summary(summary, global_step)
            train_data.reset()

        self.save()
        train_writer.close()
        print("Training complete.")

    def default(o):
        if isinstance(o, np.int64): return int(o)
        raise TypeError

    def eval(self, sess, eval_gt_coco, eval_data, vocabulary):
        """ Evaluate the model using the COCO val2014 data. """
        print("Evaluating the model ...")
        config = self.config

        results = []
        if not os.path.exists(config.eval_result_dir):
            os.mkdir(config.eval_result_dir)

        # Generate the captions for the images
        idx = 0
        for k in tqdm(list(range(eval_data.num_batches)), desc='batch'):
            batch = eval_data.next_batch()
            caption_data = self.beam_search(sess, batch, vocabulary)

            fake_cnt = 0 if k<eval_data.num_batches-1 \
                         else eval_data.fake_count
            for l in range(eval_data.batch_size-fake_cnt):
                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                caption = vocabulary.get_sentence(word_idxs)
                results.append({'image_id': int(eval_data.image_ids[idx]),
                                'caption': caption})
                idx += 1

                # Save the result in an image file, if requested
                if config.save_eval_result_as_image:
                    image_file = batch[l]
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.splitext(image_name)[0]
                    img = plt.imread(image_file)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(caption)
                    plt.savefig(os.path.join(config.eval_result_dir,
                                             image_name+'_result.jpg'))

        print("Size of result dump: ", len(results))
        fp = open(config.eval_result_file, 'w')
        json.dump(results, fp)
        fp.close()
        print("Captions written to:", config.eval_result_file)

        print("Evaluate Captions.")
        # Evaluate these captions
        eval_result_coco = eval_gt_coco.loadRes(config.eval_result_file)
        scorer = COCOEvalCap(eval_gt_coco, eval_result_coco)
        scorer.evaluate()
        print("Evaluation complete.")

    def plot_attention_maps(self, image_file, attention_map, img_caption, config):
        """ plot the attention_map on the image """
        image_name = image_file.split(os.sep)[-1]
        image_name = os.path.basename(os.path.splitext(image_file)[0])
        print(image_file)
        print("U:",len(attention_map))
        
        #attention_map[(attention_maps[0], word_scores_map_list)]
        I = plt.imread(image_file)
        I_gray = skimage.color.rgb2gray(I)
        # get some img paramaters:
        height, width = I_gray.shape
        height_block = int(height/14.)
        width_block = int(width/14.)
        plt.figure(figsize=(14, 14))

        img_caption_vector = img_caption.split(" ")
        caption_length = len(img_caption_vector)
        print("Caption Length\t: ",caption_length)
        if int(caption_length/3.) == caption_length/3.:
            no_of_rows = int(caption_length/3.)
        else:
            no_of_rows = int(caption_length/3.) + 1
        print("Num of rows\t: ", no_of_rows)


        # turn the caption into a vector of the words
        for step in range(0, caption_length): # captions / time-step
            attn_map = attention_map[step][0]
            word_score = attention_map[step][1]

            plt.subplot(no_of_rows, 3, step+1)

            attention_probs = attn_map[0][0].flatten()
            # reshape the attention_probs to shape [8,8]:
            attention_probs = np.reshape(attention_probs, (14,14))

            # convert the 8x8 attention probs map to an img of the same size as the img:
            I_att = np.zeros((height, width))
            for i in range(14):
                for j in range(14):
                    I_att[i*height_block:(i+1)*height_block, j*width_block:(j+1)*width_block] =\
                                np.ones((height_block, width_block))*attention_probs[i,j]

            # blend the grayscale img and the attention img:
            alpha = 0.97
            I_blend = alpha*I_att+(1-alpha)*I_gray
            # display the blended img:
            plt.imshow(I_blend, cmap="gray")
            plt.axis('off')
            plt.title(img_caption_vector[step], fontsize=15)
            plt.savefig(os.path.join(config.test_result_dir,
                                             image_name+'_result_att_map.jpg'), bbox_inches="tight")

            for b in range(0, 3): # beams in captions
                print("# Attend[step=",step,"]:",attention_probs.shape,",\t Words[beam=",b,"]:", word_score[0][b]) 

        plt.close()

    def test(self, sess, test_data, vocabulary):
        """ Test the model using any given images. """
        print("Testing the model ...")
        config = self.config

        if not os.path.exists(config.test_result_dir):
            os.mkdir(config.test_result_dir)

        captions = []
        scores = []
        attention_map = []
        # Generate the captions for the images
        for k in tqdm(list(range(test_data.num_batches)), desc='path'):
            batch = test_data.next_batch()
            caption_data = self.beam_search(sess, batch, vocabulary)
            #caption_data, attention_map = self.beam_search_mapattn(sess, batch, vocabulary)
            #caption_data = self.dfs_search(sess, batch, vocabulary)

            fake_cnt = 0 if k<test_data.num_batches-1 \
                         else test_data.fake_count
            for l in range(test_data.batch_size-fake_cnt):
                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                caption = vocabulary.get_sentence(word_idxs)
                captions.append(caption)
                scores.append(score)

                image_file = batch[l]
                # Save the result in an image file
                if config.save_test_result_as_image:    
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.basename(os.path.splitext(image_name)[0])
                    img = plt.imread(image_file)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(caption)
                    print('> Saving Image: ', image_file,', \t caption: ',caption, ',\t score: ', score)
                    plt.savefig(os.path.join(config.test_result_dir,
                                             image_name+'_result.jpg'))
                if config.save_test_result_as_image_attn_map and len(attention_map) > 0:
                    self.plot_attention_maps(image_file, attention_map, caption, config)

        # Save the captions to a file
        results = pd.DataFrame({'image_files':test_data.image_files,
                                'caption':captions,
                                'prob':scores})
        results.to_csv(config.test_result_file)
        print("Testing complete.")

    def beam_search(self, sess, image_files, vocabulary):
        """Use beam search to generate the captions for a batch of images."""
        # Feed in the images to get the contexts and the initial LSTM states
        config = self.config
        images = self.image_loader.load_images(image_files)
        contexts, initial_memory, initial_output = sess.run(
            [self.conv_feats, self.initial_memory, self.initial_output],
            feed_dict = {self.images: images})

        partial_caption_data = []
        complete_caption_data = []
        for k in range(config.batch_size):
            initial_beam = CaptionData(sentence = [],
                                       memory = initial_memory[k],
                                       output = initial_output[k],
                                       score = 1.0)
            partial_caption_data.append(TopN(config.beam_size))
            partial_caption_data[-1].push(initial_beam)
            complete_caption_data.append(TopN(config.beam_size))

        print("\nRun beam search") 
        for idx in range(config.max_caption_length):
            partial_caption_data_lists = []
            for k in range(config.batch_size):
                data = partial_caption_data[k].extract()
                partial_caption_data_lists.append(data)
                partial_caption_data[k].reset()

            num_steps = 1 if idx == 0 else config.beam_size
            for b in range(num_steps):
                if idx == 0:
                    last_word = np.zeros((config.batch_size), np.int32)
                else:
                    last_word = np.array([pcl[b].sentence[-1]
                                        for pcl in partial_caption_data_lists],
                                        np.int32)

                last_memory = np.array([pcl[b].memory
                                        for pcl in partial_caption_data_lists],
                                        np.float32)
                last_output = np.array([pcl[b].output
                                        for pcl in partial_caption_data_lists],
                                        np.float32)

                memory, output, scores, attention_maps = sess.run(
                    [self.memory, self.output, self.probs, self.attention_maps],
                    feed_dict = {self.contexts: contexts,
                                 self.last_word: last_word,
                                 self.last_memory: last_memory,
                                 self.last_output: last_output})
                
                # Find the beam_size most probable next words
                for k in range(config.batch_size):
                    caption_data = partial_caption_data_lists[k][b]
                    #print(b,".",k,".) ",attention_maps[k].shape," Attention: ", attention_maps[k])
                    #print(k,".) ",len(scores[k])," Probs/Scores: ", scores[k])
                    words_and_scores = list(enumerate(scores[k]))
                    words_and_scores.sort(key=lambda x: -x[1])
                    words_and_scores = words_and_scores[0:config.beam_size+1]

                    # Append each of these words to the current partial caption
                    for w, s in words_and_scores:
                        sentence = caption_data.sentence + [w]
                        score = caption_data.score * s
                        beam = CaptionData(sentence,
                                           memory[k],
                                           output[k],
                                           score)
                        if vocabulary.words[w] == '.':
                            complete_caption_data[k].push(beam)
                        else:
                            partial_caption_data[k].push(beam)

        results = []
        for k in range(config.batch_size):
            if complete_caption_data[k].size() == 0:
                complete_caption_data[k] = partial_caption_data[k]
            results.append(complete_caption_data[k].extract(sort=True))

        return results


    # This class represents a directed graph using
    # adjacency list representation
    class Graph:

        # Constructor
        def __init__(self):
            # default dictionary to store graph
            self.graph = defaultdict(list)

        # function to add an edge to graph
        def addEdge(self,u,v):
            self.graph[u].append(v)



    def beam_search_mapattn(self, sess, image_files, vocabulary):
        """Use beam search to generate the captions for a batch of images."""
        # Feed in the images to get the contexts and the initial LSTM states
        config = self.config
        images = self.image_loader.load_images(image_files)
        contexts, initial_memory, initial_output = sess.run(
            [self.conv_feats, self.initial_memory, self.initial_output],
            feed_dict = {self.images: images})

        partial_caption_data = []
        complete_caption_data = []
        for k in range(config.batch_size):
            initial_beam = CaptionData(sentence = [],
                                       memory = initial_memory[k],
                                       output = initial_output[k],
                                       score = 1.0)
            partial_caption_data.append(TopN(config.beam_size))
            partial_caption_data[-1].push(initial_beam)
            complete_caption_data.append(TopN(config.beam_size))

        attention_image_word_list = []
        print("Run beam search") 
        for idx in range(config.max_caption_length):
            partial_caption_data_lists = []
            for k in range(config.batch_size):
                data = partial_caption_data[k].extract()
                partial_caption_data_lists.append(data)
                partial_caption_data[k].reset()
            attention_map_list = []
            word_scores_map_list = []
            num_steps = 1 if idx == 0 else config.beam_size
            for b in range(num_steps):
                if idx == 0:
                    last_word = np.zeros((config.batch_size), np.int32)
                else:
                    last_word = np.array([pcl[b].sentence[-1]
                                        for pcl in partial_caption_data_lists],
                                        np.int32)

                last_memory = np.array([pcl[b].memory
                                        for pcl in partial_caption_data_lists],
                                        np.float32)
                last_output = np.array([pcl[b].output
                                        for pcl in partial_caption_data_lists],
                                        np.float32)

                memory, output, scores, attention_maps = sess.run(
                    [self.memory, self.output, self.probs, self.attention_maps],
                    feed_dict = {self.contexts: contexts,
                                 self.last_word: last_word,
                                 self.last_memory: last_memory,
                                 self.last_output: last_output})
                attention_map_list.append(attention_maps)
                # Find the beam_size most probable next words
                for k in range(config.batch_size):
                    caption_data = partial_caption_data_lists[k][b]
                    #print(b,".",k,".) ",attention_maps[k].shape," Attention: ", attention_maps[k])
                    #print(k,".) ",len(scores[k])," Probs/Scores: ", scores[k])
                    words_and_scores = list(enumerate(scores[k]))
                    words_and_scores.sort(key=lambda x: -x[1])
                    words_and_scores = words_and_scores[0:config.beam_size+1]
                    part_wordlist = []
                    # Append each of these words to the current partial caption
                    for w, s in words_and_scores:
                        sentence = caption_data.sentence + [w]
                        score = caption_data.score * s
                        beam = CaptionData(sentence,
                                           memory[k],
                                           output[k],
                                           score)
                        part_wordlist.append((w, score)) # Add the word and score tuple to list 
                        if vocabulary.words[w] == '.':
                            complete_caption_data[k].push(beam)
                        else:
                            partial_caption_data[k].push(beam)
                    word_scores_map_list.append(part_wordlist)
                #print(b,".) ",attention_maps[0].shape,", words: ", word_scores_map_list)
            attention_image_word_list.append((attention_maps, word_scores_map_list))
        results = []
        for k in range(config.batch_size):
            if complete_caption_data[k].size() == 0:
                complete_caption_data[k] = partial_caption_data[k]
            results.append(complete_caption_data[k].extract(sort=True))

        return results, attention_image_word_list

    def save(self):
        """ Save the model. """
        config = self.config
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.save_dir, str(self.global_step.eval()))

        print((" Saving the model to %s..." % (save_path+".npy")))
        np.save(save_path, data)
        info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)
        info_file.close()
        print("Model saved.")

    def load(self, sess, model_file=None):
        """ Load the model. """
        config = self.config
        if model_file is not None:
            save_path = model_file
        else:
            info_path = os.path.join(config.save_dir, "config.pickle")
            info_file = open(info_path, "rb")
            config = pickle.load(info_file)
            global_step = config.global_step
            info_file.close()
            save_path = os.path.join(config.save_dir,
                                     str(global_step)+".npy")

        print("Loading the model from %s..." %save_path)
        data_dict = np.load(save_path, encoding='latin1').item()
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("%d tensors loaded." %count)

    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading the CNN from %s..." %data_path)
        data_dict = np.load(data_path, encoding='latin1').item()
        count = 0
        for op_name in tqdm(data_dict):
            with tf.variable_scope(op_name, reuse = True):
                for param_name, data in data_dict[op_name].items():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                    except ValueError:
                        pass
        print("%d tensors loaded." %count)
