import sys

import numpy as np

if not 'texar_repo' in sys.path:
  sys.path += ['texar_repo']

from model import *
from preprocess import *


start_tokens = tf.fill([tx.utils.get_batch_size(src_input_ids)],
                       bos_token_id)
predictions = decoder(
    memory=encoder_output,
    memory_sequence_length=src_input_length,
    decoding_strategy='infer_greedy',
    beam_width=beam_width,
    alpha=alpha,
    start_tokens=start_tokens,
    end_token=eos_token_id,
    max_decoding_length=400,
    mode=tf.estimator.ModeKeys.PREDICT
)
if beam_width <= 1:
    inferred_ids = predictions[0].sample_id
else:
    # Uses the best sample by beam search
    inferred_ids = predictions['sample_id'][:, :, 0]


tokenizer = tokenization.FullTokenizer(
      vocab_file=os.path.join(bert_pretrain_dir, 'vocab.txt'),
      do_lower_case=True)


if __name__=="__main__":
    story = 'STORY'  # TODO


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    tokens_a = tokenizer.tokenize(story)
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    if len(tokens_a) > max_seq_length_src - 2:
        tokens_a = tokens_a[0:(max_seq_length_src - 2)]

    tokens_src = []
    segment_ids_src = []
    tokens_src.append("[CLS]")
    segment_ids_src.append(0)
    for token in tokens_a:
        tokens_src.append(token)
        segment_ids_src.append(0)
    tokens_src.append("[SEP]")
    segment_ids_src.append(0)

    input_ids_src = tokenizer.convert_tokens_to_ids(tokens_src)
    input_mask_src = [1] * len(input_ids_src)

    while len(input_ids_src) < max_seq_length_src:
        input_ids_src.append(0)
        input_mask_src.append(0)
        segment_ids_src.append(0)

    features = InputFeatures(src_input_ids=input_ids_src,src_input_mask=input_mask_src,src_segment_ids=segment_ids_src,
                             tgt_input_ids=None,tgt_input_mask=None,tgt_labels=None)

    feed_dict = {
        src_input_ids:np.array(features.src_input_ids).reshape(-1,1),
        src_segment_ids : np.array(features.src_segment_ids).reshape(-1,1)
    }

    hypotheses = []
    fetches = {
        'inferred_ids': inferred_ids,
    }
    fetches_ = sess.run(fetches, feed_dict=feed_dict)
    labels = np.array(features.tgt_labels).reshape(-1,1)
    hypotheses.extend(h.tolist() for h in fetches_['inferred_ids'])
    hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
    hwords = tokenizer.convert_ids_to_tokens(hypotheses[0])
    hwords = tx.utils.str_join(hwords).replace(" ##","")

    print("Original", story)
    print("Generated",hwords)
