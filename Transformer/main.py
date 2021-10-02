import tensorflow as tf
from hyperparams import *
from transformer import Transformer
from enron import create_dataset
from lossAndMetrics import *
import time
from output import *
from asistenteVirtual import AsistenteVirtual


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class ExportAV(tf.Module):
    def __init__(self, av):
        self.av = av

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        (result,
         tokens,
         attention_weights) = self.av(sentence, max_length=100)

        return result


learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

train_batches, test_batches, tokenizers = create_dataset(saved_ds = True)

transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                          input_vocab_size=tokenizers.en.get_vocab_size().numpy(),
                          target_vocab_size=tokenizers.en.get_vocab_size().numpy(), pe_input=1000,
                          pe_target=1000, rate=dropout_rate)

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp], training=True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))


for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    num_batches = train_batches.cardinality().numpy()
    print(f'Epoch {epoch + 1}:')
    print('-' * len(f'Epoch {epoch + 1}:'))
    myprint(f'Epoch {epoch + 1}:')
    myprint('-' * len(f'Epoch {epoch + 1}:'))
    # inp -> inits, tar -> body
    for (batch, (inp, tar)) in enumerate(train_batches):
        train_step(inp, tar)
        printProgressBar(batch, num_batches,
                         {'Loss': f'{train_loss.result():.4f}', 'Accuracy': f'{train_accuracy.result():.4f}'},
                         prefix='Progress:', suffix='Complete', length=50, printEnd='')
        if batch % 50 == 0:
            myprint(
                f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')
        myprint(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
    myprint(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
    myprint(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

av = AsistenteVirtual(tokenizers, transformer)
av = ExportAV(av)
tf.saved_model.save(av, export_dir='virtualAssistant')

